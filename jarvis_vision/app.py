"""Application video Jarvis : detection des vehicules, plaques et trafic."""

import argparse
from pathlib import Path
import sys

from jarvis_vision.traffic import (
    centered_traffic_region,
    count_vehicles_in_region,
    densest_region_index,
    is_traffic_saturated,
)

VEHICLE_COLORS = {
    "car": (0, 200, 0),
    "motorcycle": (0, 0, 255),
    "bus": (255, 140, 0),
    "truck": (0, 165, 255),
    "bicycle": (0, 220, 255),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HUD Jarvis de detection de vehicules et d'alerte trafic."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Index de webcam ou chemin d'une video (defaut : 0).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Poids YOLO de detection des objets (defaut : yolo11n.pt).",
    )
    parser.add_argument(
        "--output",
        help="Chemin facultatif pour enregistrer la video annotee.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Vehicules dans la zone ROI avant alerte (defaut : 4).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Confiance minimale YOLO entre 0 et 1 (defaut : 0.35).",
    )
    parser.add_argument(
        "--no-plates",
        action="store_true",
        help="Desactiver la detection et la lecture automatique des plaques.",
    )
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=8,
        help="Intervalle d'analyse OCR en images (defaut : 8).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Ne pas ouvrir de fenetre (utile pour un traitement sans ecran).",
    )
    return parser


def _source_value(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def _vehicle_class_ids(names: dict[int, str] | list[str]) -> list[int]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return [
        class_id
        for class_id, name in items
        if str(name).lower() in {*VEHICLE_COLORS, "van"}
    ]


def _read_plate(alpr, vehicle_crop) -> str | None:
    results = [
        result
        for result in alpr.predict(vehicle_crop)
        if result.ocr.text and result.ocr.text.strip()
    ]
    if not results:
        return None

    best_result = max(results, key=lambda result: result.detection.confidence)
    return best_result.ocr.text.strip().upper()


def _draw_label(cv2, frame, text: str, x: int, y: int, color: tuple[int, int, int]):
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
    )
    top = max(0, y - text_height - baseline - 8)
    right = min(frame.shape[1], x + text_width + 10)
    cv2.rectangle(frame, (x, top), (right, y), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 5, max(text_height + 2, y - baseline - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> int:
    if args.threshold < 1 or args.ocr_interval < 1:
        raise ValueError("--threshold et --ocr-interval doivent etre superieurs a 0.")
    if not 0 < args.confidence <= 1:
        raise ValueError("--confidence doit etre compris entre 0 et 1.")

    try:
        import cv2
        from ultralytics import YOLO
        if not args.no_plates:
            from fast_alpr import ALPR
    except ImportError as error:
        raise RuntimeError(
            "Dependances absentes. Installez-les avec : "
            "python -m pip install -r requirements.txt"
        ) from error

    detector = YOLO(args.model)
    vehicle_ids = _vehicle_class_ids(detector.names)
    if not vehicle_ids:
        raise ValueError("Le modele YOLO ne contient aucune classe de vehicule reconnue.")

    alpr = ALPR() if not args.no_plates else None

    capture = cv2.VideoCapture(_source_value(args.source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Impossible d'ouvrir la source video : {args.source}")

    writer = None
    frame_number = 0
    plate_cache: dict[str, str] = {}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_number += 1
            height, width = frame.shape[:2]
            region = centered_traffic_region(width, height)
            raw_frame = frame.copy() if alpr is not None else frame
            results = detector.track(
                frame,
                persist=True,
                verbose=False,
                conf=args.confidence,
                classes=vehicle_ids,
            )
            traffic_boxes: list[tuple[float, float, float, float]] = []

            result = results[0]
            if result.boxes is not None:
                for index, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                    class_id = int(box.cls.item())
                    label = str(detector.names[class_id]).lower()
                    color = VEHICLE_COLORS.get(label, (180, 0, 180))
                    confidence = float(box.conf.item())
                    traffic_boxes.append((x1, y1, x2, y2))

                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id.item())
                    cache_key = str(track_id) if track_id is not None else f"car-{index}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    _draw_label(
                        cv2,
                        frame,
                        f"{label} {confidence:.0%}",
                        x1,
                        y1,
                        color,
                    )

                    if label == "car" and alpr is not None:
                        if frame_number % args.ocr_interval == 0:
                            vehicle_crop = raw_frame[max(0, y1) : y2, max(0, x1) : x2]
                            if vehicle_crop.size:
                                plate = _read_plate(alpr, vehicle_crop)
                                if plate:
                                    plate_cache[cache_key] = plate
                        plate = plate_cache.get(cache_key)
                        if plate:
                            _draw_label(
                                cv2,
                                frame,
                                f"PLAQUE: {plate}",
                                min(max(0, x2 + 5), max(0, width - 175)),
                                min(max(y1 + 30, 20), height - 2),
                                (30, 30, 30),
                            )

            vehicle_count = count_vehicles_in_region(traffic_boxes, region)
            saturated = is_traffic_saturated(vehicle_count, args.threshold)
            cv2.rectangle(
                frame,
                (region.x1, region.y1),
                (region.x2, region.y2),
                (0, 0, 255) if saturated else (255, 200, 0),
                2,
            )
            alert = (
                "RALENTIR - GARDER SES DISTANCES"
                if saturated
                else "CIRCULATION LOCALE FLUIDE"
            )
            hud_color = (0, 0, 200) if saturated else (40, 90, 40)
            cv2.rectangle(frame, (10, 10), (min(width - 10, 470), 72), hud_color, -1)
            cv2.putText(
                frame,
                f"ZONE ROI : {vehicle_count} vehicule(s)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                alert,
                (20, 59),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if saturated:
                dense_section = densest_region_index(traffic_boxes, region)
                if dense_section is not None:
                    section_width = (region.x2 - region.x1) / 3
                    arrow_x = int(region.x1 + (dense_section + 0.5) * section_width)
                    arrow_start = (arrow_x, max(75, region.y1 - 58))
                    arrow_end = (arrow_x, region.y1 - 8)
                    cv2.arrowedLine(
                        frame,
                        arrow_start,
                        arrow_end,
                        (0, 0, 255),
                        6,
                        cv2.LINE_AA,
                        tipLength=0.4,
                    )
                    cv2.putText(
                        frame,
                        "TRAFIC DENSE",
                        (max(5, arrow_x - 75), max(65, region.y1 - 65)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if args.output and writer is None:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fps = capture.get(cv2.CAP_PROP_FPS)
                fps = fps if fps > 0 else 30.0
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Impossible de creer la video : {output_path}")

            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Jarvis - HUD Moto", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (RuntimeError, ValueError) as error:
        print(f"Erreur Jarvis : {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
