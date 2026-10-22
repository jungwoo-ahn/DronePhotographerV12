// Paper-figure deck: one slide per finished figure (PNG at 300 dpi), with what each figure shows.
//   node figures/deck/make_deck.js   ->  figures/deck/paper_figures.pptx
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { imageSize } = (() => {
  // PNG width/height from the IHDR chunk, so images keep their aspect ratio
  return {
    imageSize: (f) => {
      const b = fs.readFileSync(f);
      return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
    },
  };
})();
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const FIG = path.resolve(__dirname, "..");
const THEME = {
  name: "Shot Profile Figures",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2126", lt1: "FFFFFF", dk2: "165A3C", lt2: "F3F4F6",
    accent1: "22855A", accent2: "2E5FBF", accent3: "D06A1E", accent4: "6A6EA0",
    accent5: "C4553C", accent6: "D99A26", hlink: "2E5FBF", folHlink: "6A6EA0",
  },
};

const SLIDES = [
  {
    section: "Method",
    title: "Fig. 2 — Method overview",
    img: "overview/fig2_overview.png",
    points: [
      ["Data engine", "random camera orbits around a placed subject; any frame a flight reaches is a goal, labelled from the projected mesh"],
      ["World camera action model", "frozen Qwen3-VL expert + trained world-action expert in one joint attention; 8 × 10-D action tokens (Δt, R6D, shoot)"],
      ["Deployment", "intent → LLM → shot profile → closed-loop chunks until the model declares shoot"],
    ],
    notes: "figures/overview/make_fig2.py -> fig2_overview.{pdf,svg,png}. Rollout in (c): All-People-Are-Sisters p00 s31 g02, d 2.01 -> 0.066, shoot at step 7 of chunk 4. The user sentence in (c) is illustrative.",
  },
  {
    section: "Method",
    title: "Data-generation pipeline",
    img: "datagen/fig_datagen.png",
    points: [
      ["Funnel", "9,889 scene × subject runs → 7,885 placements → 393k camera pairs → 75k trajectories → 2.4M frames → 150k windows"],
      ["Placement", "Gemini interest region, ray-cast floor cells, VLM judge keeps a spot at quality ≥ 7"],
      ["Trajectory", "ellipse centred at the look-at O, C_far as its vertex, through C_near; 32 frames at equal Δθ"],
      ["Panel (b)", "schematic grid in this version"],
    ],
    notes: "figures/datagen/make_fig_datagen.py -> fig_datagen.*. Counts were taken from the data on 2026-10-14.",
  },
  {
    section: "Method",
    title: "Data-generation pipeline — real floor grid",
    img: "datagen/fig_datagen_realgrid.png",
    points: [
      ["Panel (b)", "a real re-run of the floor search on Chill-Camping: 2,054 flat & visible cells projected into the scene camera's view"],
      ["VLM judge", "the placement run's real view scores 3, 2, 1, 8, 4; best 8 ≥ 7 kept"],
      ["Caveat", "re-run without the Gemini interest region, so two of the top-3 land on the lake surface"],
    ],
    notes: "figures/datagen/make_fig_datagen.py --b real -> fig_datagen_realgrid.*. Grid from figures/datagen/sbatch_placement_grid.sh (job 192374), runs/placement_grid/Chill-Camping__Knit-and-Denim.",
  },
  {
    section: "Results",
    title: "Qualitative rollouts (1/2)",
    img: "qualitative/fig_rollouts_part1.png",
    points: [
      ["Setting", "benchmark v2, same start and goal for all policies; GT goal camera as a blue star"],
      ["Ours", "d 0.09 / 0.07 / 0.02 / 0.08 — every baseline misses (d ≥ 0.11)"],
      ["Read", "3-D paths on the left, last frames and d on the right; green badge = success"],
    ],
    notes: "figures/qualitative/make_fig_rollouts.py --rows 0,1,2,4. Selected examples (Ours succeeds, pi0.5 / GR00T / T2A / DP all miss); image-to-action left out because it is given the GT goal image.",
  },
  {
    section: "Results",
    title: "Qualitative rollouts (2/2)",
    img: "qualitative/fig_rollouts_part2.png",
    points: [
      ["Scenes", "office atrium, lake shore, modern office, mountain terrace"],
      ["Ours", "d 0.05 / 0.05 / 0.07 / 0.09"],
      ["Caption note", "selected examples; say so in the paper"],
    ],
    notes: "figures/qualitative/make_fig_rollouts.py --rows 3,5,6,7.",
  },
];

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
  pres.title = "Paper figures";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "Title",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.5, w: 11.7, h: 1.3, fontSize: 44, bold: true,
                                   color: C.background1, valign: "bottom" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 1.0, fontSize: 18,
                                   color: C.background2, valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "Figure",
    background: { color: C.background1 },
    margin: [0.5, 0.5, 0.5, 0.5],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.3, h: 0.7, fontSize: 30, bold: true,
                                   color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.4, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: "61656D" },
  });

  pres.addSection({ title: "Cover" });
  const s0 = pres.addSlide({ masterName: "Title", sectionTitle: "Cover" });
  s0.addText("Paper figures — draft", { placeholder: "title" });
  s0.addText("Overview · data generation · qualitative rollouts", { placeholder: "body" });

  let current = null;
  for (const S of SLIDES) {
    if (S.section !== current) {
      pres.addSection({ title: S.section });
      current = S.section;
    }
    const s = pres.addSlide({ masterName: "Figure", sectionTitle: S.section });
    s.addText(S.title, { placeholder: "title" });
    // figure: fit into a 9.0 x 6.2 in box at the left
    const f = path.join(FIG, S.img);
    const { w, h } = imageSize(f);
    const bw = 9.0, bh = 6.2;
    const k = Math.min(bw / w, bh / h);
    const iw = w * k, ih = h * k;
    s.addImage({ path: f, x: 0.5, y: 1.15 + (bh - ih) / 2, w: iw, h: ih, objectName: "figure",
                 altText: S.title });
    // takeaways in a tinted panel on the right
    const px = 0.5 + iw + 0.35, pw = 13.33 - 0.5 - px;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px, y: 1.15, w: pw, h: 6.2, fill: { color: C.background2 },
                                                line: { type: "none" }, rectRadius: 0.12, objectName: "notes panel" });
    const runs = [];
    S.points.forEach(([head, body], i) => {
      runs.push({ text: head, options: { bold: true, color: C.accent1, fontSize: 15, breakLine: true } });
      runs.push({ text: body, options: { color: C.text1, fontSize: 14, breakLine: i < S.points.length - 1,
                                         paraSpaceAfter: 14 } });
    });
    s.addText(runs, { x: px + 0.25, y: 1.4, w: pw - 0.5, h: 5.7, valign: "top", isTextBox: true, margin: 0,
                      objectName: "takeaways" });
    s.addNotes(S.notes);
  }

  const out = path.join(__dirname, "paper_figures.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
