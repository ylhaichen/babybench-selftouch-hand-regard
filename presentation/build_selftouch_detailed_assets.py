import csv
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = ROOT / "presentation"
ASSETS_DIR = PRESENTATION_DIR / "assets_detailed"

BASE_METRICS = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_smooth" / "live_metrics.csv"
DIFFICULT_METRICS = ROOT / "results_author_replica" / "resume_random_init" / "live_metrics.csv"
AFTER_METRICS = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_after_difficult_task" / "live_metrics.csv"

DIFFICULT_PLOT = ROOT / "results_author_replica" / "resume_random_init" / "live_metrics.png"
AFTER_PLOT = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_after_difficult_task" / "live_metrics.png"

EVAL0_VIDEO = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_after_difficult_task" / "videos" / "evaluation_0.avi"
EVAL8_VIDEO = ROOT / "results_author_replica" / "intrinsic_motivation_stelios_x_giannis_after_difficult_task" / "videos" / "evaluation_8.avi"

AFTER_FINAL_SCORE = 0.2206
DIFFICULT_FINAL_SCORE = 0.0588


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_font(size: int, bold: bool = False):
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


TITLE_FONT = load_font(42, bold=True)
SUBTITLE_FONT = load_font(26, bold=True)
BODY_FONT = load_font(22)
SMALL_FONT = load_font(18)


def draw_box(draw: ImageDraw.ImageDraw, box, title: str, body: str, fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=3)
    x0, y0, _, _ = box
    draw.text((x0 + 22, y0 + 16), title, font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text((x0 + 22, y0 + 58), body, font=BODY_FONT, fill="#24374d", spacing=7)


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, fill="#355c7d") -> None:
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


def crop_panel(path: Path, box, out_name: str) -> Path:
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
    out = ASSETS_DIR / out_name
    Image.fromarray(frame).save(out)
    return out


def read_last_metrics(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def metric_row(name: str, row: dict[str, str]) -> tuple[str, str, str, str]:
    steps = float(row["timesteps"]) / 1_000_000.0
    ret = float(row["episode_return_mean"])
    left = int(float(row["global_left_geoms"]))
    right = int(float(row["global_right_geoms"]))
    return (name, f"{steps:.1f}M", f"{ret:.2f}", f"{left} / {right}")


def slide_canvas(title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), "#fbfcff")
    draw = ImageDraw.Draw(image)
    draw.text((70, 44), title, font=TITLE_FONT, fill="#0f2744")
    if subtitle:
        draw.text((72, 106), subtitle, font=SMALL_FONT, fill="#5a6d84")
    return image, draw


def save_slide1_cover() -> Path:
    image, draw = slide_canvas(
        "Learning Infant-Like Self-Touch in BabyBench",
        "Detailed coursework deck with expanded motivation and results discussion",
    )
    hero = Image.open(extract_frame(EVAL0_VIDEO, 240, "slide1_hero_temp.png")).convert("RGB").resize((760, 506))
    image.paste(hero, (770, 180))
    draw.rounded_rectangle((758, 168, 1544, 700), radius=24, outline="#b8cce4", width=3, fill=None)

    draw.multiline_text(
        (85, 230),
        "Task\n\nSelf-touch learning without extrinsic rewards\n\nPlatform\n\nBabyBench + MIMo\n\nApproach\n\nIntrinsic-motivation reinforcement learning",
        font=BODY_FONT,
        fill="#24374d",
        spacing=11,
    )
    draw.multiline_text(
        (85, 700),
        "Research question:\nCan an infant-like robot discover its own body through self-generated exploration,\nusing tactile and proprioceptive feedback only?",
        font=SMALL_FONT,
        fill="#40556e",
        spacing=8,
    )
    out = ASSETS_DIR / "slide1_cover_detailed.png"
    image.save(out)
    return out


def save_slide2_motivation() -> Path:
    image, draw = slide_canvas(
        "Motivation 1: what infant self-touch studies suggest",
        "Framed around developmental psychology rather than a robotics replication claim",
    )
    boxes = [
        (
            (70, 180, 760, 390),
            "Early spontaneous behaviour",
            "Infants begin touching their own bodies very early in life.\nThese movements are not externally instructed,\nbut they generate rich sensorimotor experience.",
            "#eaf4ff",
            "#8fb4d8",
        ),
        (
            (830, 180, 1520, 390),
            "Dual sensory feedback",
            "Each self-touch event combines proprioceptive information\nfrom movement with tactile information from contact,\ncreating a natural learning signal for body awareness.",
            "#eef7ea",
            "#96c18d",
        ),
        (
            (70, 440, 760, 710),
            "Developmental relevance",
            "Developmental studies connect self-touch with the emergence\nof body schema, self-perception, and later sensorimotor coordination.",
            "#fff3e6",
            "#d3a36d",
        ),
        (
            (830, 440, 1520, 710),
            "Project-level inspiration",
            "The project uses the same basic developmental idea:\nlet the agent learn from self-generated interaction,\ninstead of giving it an external task-specific reward.",
            "#f7ecff",
            "#b691d1",
        ),
    ]
    for box, title, body, fill, outline in boxes:
        draw_box(draw, box, title, body, fill, outline)

    draw.multiline_text(
        (70, 770),
        "Suggested citations: Thomas et al. (2015), Rochat (1998), and DiMercurio et al. (2018).",
        font=SMALL_FONT,
        fill="#5a6d84",
        spacing=6,
    )
    out = ASSETS_DIR / "slide2_motivation_detailed.png"
    image.save(out)
    return out


def save_slide3_motivation_robotics() -> Path:
    image, draw = slide_canvas(
        "Motivation 2: why this is a meaningful robotics problem",
        "From infant self-exploration to intrinsic-motivation reinforcement learning",
    )
    left = (70, 185, 590, 720)
    right = (1010, 185, 1530, 720)
    center = (650, 250, 950, 650)

    draw_box(
        draw,
        left,
        "Developmental insight",
        "Self-touch is useful because it is:\n\n• self-generated\n• multimodal\n• body-centered\n• available without labels\n\nThat makes it a natural example of autonomous learning.",
        "#eaf4ff",
        "#8fb4d8",
    )
    draw_box(
        draw,
        right,
        "Robotics translation",
        "In robotics, this suggests a practical hypothesis:\n\nA policy may learn early body knowledge if it is rewarded for novel,\nmeaningful self-contact rather than an externally defined task target.",
        "#eef7ea",
        "#96c18d",
    )
    draw.rounded_rectangle(center, radius=26, fill="#fff7f0", outline="#d6a26e", width=3)
    draw.text((680, 278), "Why BabyBench?", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (680, 338),
        "• infant-like embodiment\n• tactile + proprioceptive sensing\n• continuous motor control\n• benchmarked evaluation\n• suitable for intrinsic-motivation studies",
        font=BODY_FONT,
        fill="#24374d",
        spacing=10,
    )
    draw_arrow(draw, (590, 450), (650, 450))
    draw_arrow(draw, (950, 450), (1010, 450))

    out = ASSETS_DIR / "slide3_robotics_motivation.png"
    image.save(out)
    return out


def save_slide4_method() -> Path:
    image, draw = slide_canvas(
        "Method: intrinsic-motivation RL pipeline",
        "Implementation details tailored to the BabyBench self-touch benchmark",
    )
    boxes = [
        ((50, 215, 310, 450), "Observations", "Touch\n+\nProprioception", "#eaf4ff", "#8fb4d8"),
        ((360, 215, 680, 450), "Feature representation", "Raw tactile input is compressed\ninto 68 grouped touch features\nbefore being fused with proprioception.", "#eef7ea", "#96c18d"),
        ((730, 215, 1040, 450), "Intrinsic reward", "Novelty\nCoverage\nMilestones\nBalance", "#fff3e6", "#d3a36d"),
        ((1090, 215, 1360, 450), "Policy learning", "PPO learns a continuous motor policy\nfrom the intrinsic signal.", "#f7ecff", "#b691d1"),
    ]
    for box, title, body, fill, outline in boxes:
        draw_box(draw, box, title, body, fill, outline)
    draw_arrow(draw, (310, 333), (360, 333))
    draw_arrow(draw, (680, 333), (730, 333))
    draw_arrow(draw, (1040, 333), (1090, 333))

    curriculum = (310, 555, 1360, 790)
    draw.rounded_rectangle(curriculum, radius=26, fill="#fff7f0", outline="#d6a26e", width=3)
    draw.text((340, 585), "Training curriculum and model selection", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (340, 645),
        "Base stage (5M): standard resets for initial self-touch learning\n\n"
        "Difficult stage (2M): randomized initial joints for broader exploration\n\n"
        "After stage (2M): final refinement, then select the checkpoint by standard evaluation performance",
        font=BODY_FONT,
        fill="#24374d",
        spacing=10,
    )
    out = ASSETS_DIR / "slide4_method_detailed.png"
    image.save(out)
    return out


def save_slide5_algorithm() -> Path:
    image, draw = slide_canvas(
        "Detailed algorithmic design",
        "How observation processing, intrinsic reward computation, and PPO interact at each environment step",
    )
    left_top = (60, 180, 500, 425)
    left_bottom = (60, 470, 500, 790)
    center = (560, 180, 1025, 790)
    right = (1085, 180, 1540, 790)

    draw.rounded_rectangle(left_top, radius=26, fill="#eaf4ff", outline="#8fb4d8", width=3)
    draw.text((80, 200), "1. Observation encoding", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (80, 250),
        "Touch and proprioception\ncome from the Dict observation.\n\n"
        "Touch is flattened,\nconverted to sensor magnitudes,\nand average-pooled into\n68 grouped tactile features.\n\n"
        "Proprioception remains\na separate input branch.",
        font=SMALL_FONT,
        fill="#24374d",
        spacing=9,
    )

    draw.rounded_rectangle(left_bottom, radius=26, fill="#eef7ea", outline="#96c18d", width=3)
    draw.text((80, 490), "2. Intrinsic reward bookkeeping", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (80, 540),
        "The wrapper tracks:\n\n"
        "• part novelty\n"
        "• voxel novelty\n"
        "• hand-body contact novelty\n"
        "• milestone bonuses\n"
        "• left-right balance bonus\n\n"
        "These signals are combined into the step reward.",
        font=SMALL_FONT,
        fill="#24374d",
        spacing=8,
    )

    draw.rounded_rectangle(center, radius=26, fill="#fff7f0", outline="#d6a26e", width=3)
    draw.text((590, 205), "3. Per-step learning loop", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (590, 255),
        "1. Reset environment and read touch + proprioception\n\n"
        "2. Encode observation into policy features\n\n"
        "3. PPO policy outputs a continuous action\n\n"
        "4. Step the BabyBench / MIMo environment\n\n"
        "5. Scan touch activation and MuJoCo contacts\n\n"
        "6. Compute intrinsic reward and update statistics\n\n"
        "7. Store the transition in the PPO rollout buffer\n\n"
        "8. Periodically optimize the actor and critic",
        font=SMALL_FONT,
        fill="#24374d",
        spacing=8,
    )

    draw.rounded_rectangle(right, radius=26, fill="#f7ecff", outline="#b691d1", width=3)
    draw.text((1115, 205), "4. Curriculum logic", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (1115, 255),
        "Base stage\n"
        "5M steps, standard resets\n\n"
        "Difficult stage\n"
        "2M steps, random-initialized joints\nfor broader exploration\n\n"
        "After stage\n"
        "2M steps, refinement under\nmore standard resets\n\n"
        "Final model\n"
        "selected by deterministic evaluation,\nnot by training coverage alone",
        font=SMALL_FONT,
        fill="#24374d",
        spacing=8,
    )

    draw_arrow(draw, (500, 300), (560, 300))
    draw_arrow(draw, (500, 610), (560, 610))
    draw_arrow(draw, (1025, 485), (1085, 485))

    out = ASSETS_DIR / "slide5_algorithm_detailed.png"
    image.save(out)
    return out


def save_slide6_training() -> Path:
    crop_panel(DIFFICULT_PLOT, (1100, 25, 2190, 675), "slide6_difficult_coverage_panel.png")
    crop_panel(AFTER_PLOT, (35, 690, 1090, 1360), "slide6_after_episode_summary_panel.png")

    difficult_cov = Image.open(ASSETS_DIR / "slide6_difficult_coverage_panel.png").convert("RGB").resize((760, 420))
    after_ep = Image.open(ASSETS_DIR / "slide6_after_episode_summary_panel.png").convert("RGB").resize((760, 420))

    image, draw = slide_canvas(
        "Training progress across the three stages",
        "Use the plots to explain what improved and what still remained limited",
    )
    image.paste(difficult_cov, (60, 170))
    image.paste(after_ep, (60, 520))

    draw.text((890, 175), "How to explain these plots", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (890, 225),
        "• Base stage learned stable contact but limited body coverage.\n\n"
        "• Difficult stage increased exploratory contact strongly under randomized initial states.\n\n"
        "• After stage stabilized the policy again and was chosen as the final model under deterministic evaluation.",
        font=BODY_FONT,
        fill="#24374d",
        spacing=10,
    )

    summary_box = (890, 520, 1510, 835)
    draw.rounded_rectangle(summary_box, radius=24, fill="#ffffff", outline="#c8d5e5", width=3)
    draw.text((920, 548), "Stage summary", font=SUBTITLE_FONT, fill="#10213a")
    headers = ["Stage", "Steps", "Return", "L / R geoms"]
    xs = [920, 1080, 1215, 1360]
    for x, header in zip(xs, headers):
        draw.text((x, 605), header, font=SMALL_FONT, fill="#597089")
    for row_y, row in zip(
        [660, 725, 790],
        [
            metric_row("Base", read_last_metrics(BASE_METRICS)),
            metric_row("Difficult", read_last_metrics(DIFFICULT_METRICS)),
            metric_row("After", read_last_metrics(AFTER_METRICS)),
        ],
    ):
        for x, value in zip(xs, row):
            draw.text((x, row_y), value, font=BODY_FONT, fill="#24374d")

    out = ASSETS_DIR / "slide6_training_detailed.png"
    image.save(out)
    return out


def save_slide7_results() -> Path:
    demo = Image.open(extract_frame(EVAL0_VIDEO, 540, "slide7_demo_frame.png")).convert("RGB").resize((700, 466))
    alt = Image.open(extract_frame(EVAL8_VIDEO, 480, "slide7_demo_frame_alt.png")).convert("RGB").resize((250, 166))
    image, draw = slide_canvas(
        "Final evaluation, interpretation, and takeaway",
        "The selected model is the after-stage checkpoint because it generalizes better under standard evaluation",
    )
    image.paste(demo, (60, 190))
    image.paste(alt, (540, 500))

    summary = (830, 190, 1520, 780)
    draw.rounded_rectangle(summary, radius=24, fill="#ffffff", outline="#c8d5e5", width=3)
    draw.text((860, 220), "Evaluation summary", font=SUBTITLE_FONT, fill="#10213a")
    draw.multiline_text(
        (860, 278),
        "Final selected model: after-stage final checkpoint\n"
        "Evaluation protocol: 10 episodes, deterministic policy\n"
        f"After-stage manual score: {AFTER_FINAL_SCORE:.4f}\n"
        f"Difficult-stage manual score: {DIFFICULT_FINAL_SCORE:.4f}\n"
        "After-stage touched geoms: left 9, right 6\n\n"
        "Interpretation:\n"
        "The policy demonstrates meaningful autonomous self-exploration,\n"
        "but body coverage and generalization are still limited.\n\n"
        "Future work:\n"
        "better tactile representation, stronger generalization,\n"
        "and more robust evaluation performance.",
        font=BODY_FONT,
        fill="#24374d",
        spacing=9,
    )
    out = ASSETS_DIR / "slide7_results_detailed.png"
    image.save(out)
    return out


def main() -> None:
    ensure_dirs()
    save_slide1_cover()
    save_slide2_motivation()
    save_slide3_motivation_robotics()
    save_slide4_method()
    save_slide5_algorithm()
    save_slide6_training()
    save_slide7_results()
    print(f"Generated detailed presentation assets in: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
