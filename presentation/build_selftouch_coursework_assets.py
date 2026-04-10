import csv
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = ROOT / "presentation"
ASSETS_DIR = PRESENTATION_DIR / "assets"

BASE_METRICS = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_smooth" / "live_metrics.csv"
DIFFICULT_METRICS = ROOT / "results_author_replica" / "resume_random_init" / "live_metrics.csv"
AFTER_METRICS = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_after_difficult_task" / "live_metrics.csv"

DIFFICULT_PLOT = ROOT / "results_author_replica" / "resume_random_init" / "live_metrics.png"
AFTER_PLOT = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_after_difficult_task" / "live_metrics.png"

EVAL0_VIDEO = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_after_difficult_task" / "videos" / "evaluation_0.avi"
EVAL8_VIDEO = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_after_difficult_task" / "videos" / "evaluation_8.avi"


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(44, bold=True)
SUBTITLE_FONT = load_font(28, bold=True)
BODY_FONT = load_font(24)
SMALL_FONT = load_font(20)


def read_last_metrics(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def format_metric_row(name: str, row: dict[str, str]) -> tuple[str, str, str, str]:
    steps = float(row["timesteps"]) / 1_000_000.0
    ret = float(row["episode_return_mean"])
    left = int(float(row["global_left_geoms"]))
    right = int(float(row["global_right_geoms"]))
    return (name, f"{steps:.1f}M", f"{ret:.2f}", f"{left} / {right}")


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - text_w) / 2
    y = y0 + (y1 - y0 - text_h) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=8, align="center")


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str, fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=3)
    x0, y0, x1, _ = box
    draw.text((x0 + 22, y0 + 18), title, font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text((x0 + 22, y0 + 64), body, font=BODY_FONT, fill="#24374d", spacing=8)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#355c7d") -> None:
    draw.line([start, end], fill=fill, width=7)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 18
    left = (
        end[0] - size * math.cos(angle) + size * 0.6 * math.sin(angle),
        end[1] - size * math.sin(angle) - size * 0.6 * math.cos(angle),
    )
    right = (
        end[0] - size * math.cos(angle) - size * 0.6 * math.sin(angle),
        end[1] - size * math.sin(angle) + size * 0.6 * math.cos(angle),
    )
    draw.polygon([end, left, right], fill=fill)


def save_concept_graphic() -> Path:
    image = Image.new("RGB", (1600, 900), "#fbfcff")
    draw = ImageDraw.Draw(image)
    draw.text((80, 56), "Infant self-touch as a developmental loop", font=TITLE_FONT, fill="#0f2744")
    draw.text(
        (80, 120),
        "A simple concept figure for Slide 2",
        font=SMALL_FONT,
        fill="#5a6d84",
    )

    boxes = [
        ((90, 280, 470, 560), "Movement", "Spontaneous arm and hand\nmovements generate\nself-contact opportunities.", "#eaf4ff", "#8fb4d8"),
        ((610, 280, 990, 560), "Tactile feedback", "Touch signals indicate where\ncontact happened on\nthe body surface.", "#eef7ea", "#96c18d"),
        ((1130, 280, 1510, 560), "Body awareness", "Repeated sensorimotor\nexperience supports an\nemerging body schema.", "#fff3e6", "#d3a36d"),
    ]
    for box, title, body, fill, outline in boxes:
        draw_box(draw, box, title, body, fill, outline)

    draw_arrow(draw, (470, 420), (610, 420))
    draw_arrow(draw, (990, 420), (1130, 420))

    caption = (
        "Motivation paper: Thomas, Karl & Whishaw (2015)\n"
        "\"Independent development of the reach and the grasp in spontaneous self-touching by human infants in the first 6 months.\""
    )
    draw.multiline_text((90, 700), caption, font=SMALL_FONT, fill="#40556e", spacing=8)

    out = ASSETS_DIR / "slide2_concept_graphic.png"
    image.save(out)
    return out


def save_pipeline_graphic() -> Path:
    image = Image.new("RGB", (1600, 900), "#fbfcff")
    draw = ImageDraw.Draw(image)
    draw.text((80, 56), "Intrinsic-motivation RL pipeline in BabyBench", font=TITLE_FONT, fill="#0f2744")
    draw.text((80, 120), "A simple method overview for Slide 3", font=SMALL_FONT, fill="#5a6d84")

    top_boxes = [
        ((70, 245, 350, 500), "Observations", "Touch\n+\nProprioception", "#eaf4ff", "#8fb4d8"),
        ((410, 245, 710, 500), "Feature extraction", "68-group touch\ncompression +\nproprio features", "#eef7ea", "#96c18d"),
        ((770, 245, 1080, 500), "Intrinsic reward", "Novelty\nCoverage\nMilestones\nBalance", "#fff3e6", "#d3a36d"),
        ((1140, 245, 1420, 500), "Policy learning", "PPO policy\noutputs motor\nactions", "#f7ecff", "#b691d1"),
    ]
    for box, title, body, fill, outline in top_boxes:
        draw_box(draw, box, title, body, fill, outline)

    draw_arrow(draw, (350, 372), (410, 372))
    draw_arrow(draw, (710, 372), (770, 372))
    draw_arrow(draw, (1080, 372), (1140, 372))

    curriculum_box = (350, 620, 1160, 800)
    draw.rounded_rectangle(curriculum_box, radius=26, fill="#fff7f0", outline="#d6a26e", width=3)
    draw.text((380, 648), "Training curriculum", font=SUBTITLE_FONT, fill="#10213a")
    draw.text(
        (380, 700),
        "Base stage (5M)  ->  Difficult stage (2M, randomized initial joints)  ->  After stage (2M)",
        font=BODY_FONT,
        fill="#24374d",
    )

    draw_arrow(draw, (780, 500), (780, 620), fill="#a56d3b")

    out = ASSETS_DIR / "slide3_pipeline_graphic.png"
    image.save(out)
    return out


def crop_panel(path: Path, box: tuple[int, int, int, int], out_name: str) -> Path:
    image = Image.open(path).convert("RGB")
    cropped = image.crop(box)
    out = ASSETS_DIR / out_name
    cropped.save(out)
    return out


def extract_frame(video_path: Path, frame_index: int, out_name: str) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    out = ASSETS_DIR / out_name
    image.save(out)
    return out


def save_demo_contact_sheet(frames: list[tuple[Path, int, str]], out_name: str) -> Path:
    thumbs = []
    labels = []
    for video_path, frame_idx, label in frames:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).resize((420, 280))
        thumbs.append(image)
        labels.append(label)

    sheet = Image.new("RGB", (1400, 780), "#fbfcff")
    draw = ImageDraw.Draw(sheet)
    draw.text((70, 40), "Demo frame candidates from evaluation videos", font=TITLE_FONT, fill="#0f2744")
    for idx, thumb in enumerate(thumbs):
        row = idx // 3
        col = idx % 3
        x = 70 + col * 440
        y = 150 + row * 320
        sheet.paste(thumb, (x, y))
        draw.text((x, y + 292), labels[idx], font=SMALL_FONT, fill="#40556e")
    out = ASSETS_DIR / out_name
    sheet.save(out)
    return out


def save_training_summary_figure() -> Path:
    difficult_cov = Image.open(ASSETS_DIR / "slide4_difficult_coverage_panel.png").convert("RGB")
    after_ep = Image.open(ASSETS_DIR / "slide4_after_episode_summary_panel.png").convert("RGB")
    image = Image.new("RGB", (1800, 1000), "#fbfcff")
    draw = ImageDraw.Draw(image)
    draw.text((70, 50), "Training progress across the three stages", font=TITLE_FONT, fill="#0f2744")

    difficult_cov = difficult_cov.resize((780, 430))
    after_ep = after_ep.resize((780, 430))
    image.paste(difficult_cov, (60, 150))
    image.paste(after_ep, (60, 540))

    draw.text((900, 170), "Plot A: Difficult-stage coverage", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (900, 215),
        "Randomized initial states increased exploratory body contact.\nThis stage produced the strongest geometric coverage during training.",
        font=BODY_FONT,
        fill="#24374d",
        spacing=8,
    )
    draw.text((900, 355), "Plot B: After-stage episode summary", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (900, 400),
        "The final stage stabilized the policy under more standard conditions.\nThis checkpoint performed best under deterministic evaluation.",
        font=BODY_FONT,
        fill="#24374d",
        spacing=8,
    )

    summary_box = (900, 560, 1710, 900)
    draw.rounded_rectangle(summary_box, radius=26, fill="#ffffff", outline="#c8d5e5", width=3)
    draw.text((930, 585), "Stage summary", font=SUBTITLE_FONT, fill="#10213a")
    headers = ["Stage", "Steps", "Return", "L / R geoms"]
    xs = [930, 1130, 1280, 1460]
    y = 650
    for x, header in zip(xs, headers):
        draw.text((x, y), header, font=SMALL_FONT, fill="#597089")
    rows = [
        format_metric_row("Base", read_last_metrics(BASE_METRICS)),
        format_metric_row("Difficult", read_last_metrics(DIFFICULT_METRICS)),
        format_metric_row("After", read_last_metrics(AFTER_METRICS)),
    ]
    row_y = 710
    for row in rows:
        for x, value in zip(xs, row):
            draw.text((x, row_y), value, font=BODY_FONT, fill="#24374d")
        row_y += 72

    out = ASSETS_DIR / "slide4_training_progress.png"
    image.save(out)
    return out


def save_final_results_figure() -> Path:
    demo = Image.open(ASSETS_DIR / "slide5_demo_frame.png").convert("RGB").resize((880, 586))
    image = Image.new("RGB", (1800, 1000), "#fbfcff")
    draw = ImageDraw.Draw(image)
    draw.text((70, 50), "Final evaluation and takeaway", font=TITLE_FONT, fill="#0f2744")
    image.paste(demo, (60, 190))

    box = (990, 190, 1720, 780)
    draw.rounded_rectangle(box, radius=26, fill="#ffffff", outline="#c8d5e5", width=3)
    draw.text((1030, 225), "Final result summary", font=SUBTITLE_FONT, fill="#10213a")
    lines = [
        "Selected model: after-stage final checkpoint",
        "Evaluation: 10 episodes, deterministic policy",
        "Manual self-touch score: 0.2206",
        "Left touched geoms: 9",
        "Right touched geoms: 6",
        "",
        "Interpretation:",
        "The policy shows meaningful autonomous body exploration,",
        "but coverage and generalization are still limited.",
    ]
    draw.multiline_text((1030, 295), "\n".join(lines), font=BODY_FONT, fill="#24374d", spacing=10)
    draw.multiline_text(
        (60, 810),
        "Suggested use: insert this screenshot or embed evaluation_0.avi directly in Slide 5.",
        font=SMALL_FONT,
        fill="#5a6d84",
    )

    out = ASSETS_DIR / "slide5_final_results.png"
    image.save(out)
    return out


def main() -> None:
    ensure_dirs()

    save_concept_graphic()
    save_pipeline_graphic()

    crop_panel(
        DIFFICULT_PLOT,
        (1100, 25, 2190, 675),
        "slide4_difficult_coverage_panel.png",
    )
    crop_panel(
        AFTER_PLOT,
        (35, 690, 1090, 1360),
        "slide4_after_episode_summary_panel.png",
    )

    extract_frame(EVAL0_VIDEO, 240, "slide1_hero_frame.png")
    extract_frame(EVAL0_VIDEO, 540, "slide5_demo_frame.png")
    extract_frame(EVAL8_VIDEO, 480, "slide5_demo_frame_alt.png")
    save_demo_contact_sheet(
        [
            (EVAL0_VIDEO, 120, "evaluation_0 frame 120"),
            (EVAL0_VIDEO, 240, "evaluation_0 frame 240"),
            (EVAL0_VIDEO, 540, "evaluation_0 frame 540"),
            (EVAL8_VIDEO, 180, "evaluation_8 frame 180"),
            (EVAL8_VIDEO, 480, "evaluation_8 frame 480"),
            (EVAL8_VIDEO, 780, "evaluation_8 frame 780"),
        ],
        "demo_frame_candidates.png",
    )

    save_training_summary_figure()
    save_final_results_figure()

    print(f"Generated presentation assets in: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
