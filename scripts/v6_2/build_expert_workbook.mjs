import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = path.resolve(import.meta.dirname, "../..");
const artifactDir = path.join(
  repoRoot,
  "artifacts",
  "v6_2_final_validation",
  "expert_evaluation",
);
const outputDir = path.join(repoRoot, "outputs", "v6_2_expert_review");
const renderDir = path.join(outputDir, "renders");
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(renderDir, { recursive: true });

const palette = {
  navy: "#17365D",
  blue: "#D9EAF7",
  green: "#E2F0D9",
  amber: "#FFF2CC",
  red: "#FCE4D6",
  white: "#FFFFFF",
  ink: "#1F2937",
  border: "#B8C2CC",
};

function styleTable(sheet, rangeAddress, headerAddress, widths = {}) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getRange(rangeAddress);
  used.format = {
    font: { name: "Aptos", size: 10, color: palette.ink },
    verticalAlignment: "top",
  };
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#E5E7EB" },
    bottom: { style: "thin", color: palette.border },
  };
  const header = sheet.getRange(headerAddress);
  header.format = {
    fill: palette.navy,
    font: { name: "Aptos Display", size: 10, bold: true, color: palette.white },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: palette.navy },
  };
  header.format.rowHeight = 34;
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
}

function addInstructions(workbook) {
  const sheet = workbook.worksheets.add("Instructions");
  sheet.showGridLines = false;
  sheet.getRange("A1:H2").merge();
  sheet.getRange("A1").values = [["V6.2 Blinded Expert Review"]];
  sheet.getRange("A1:H2").format = {
    fill: palette.navy,
    font: { name: "Aptos Display", size: 20, bold: true, color: palette.white },
    verticalAlignment: "center",
  };
  const rows = [
    ["Purpose", "Assess technical relevance, evidence support, safety, workload, and escalation. This is not a causal effectiveness study."],
    ["Blinding", "Model identity, exact probabilities, source/student IDs, outcomes, and demographics are withheld."],
    ["Independence", "Complete your randomized order independently before discussing cases with another reviewer."],
    ["Q1", "Overall plan score: integer 1 (very poor) to 5 (very good)."],
    ["Q2", "For every proposed action: APPROVE, PARTIAL, UNSURE, or REJECT."],
    ["Q3", "Missing action: YES/NO. If YES, supply the missing-action text."],
    ["Q4", "Escalation: CORRECT, OVER_ESCALATED, UNDER_ESCALATED, or UNSURE."],
    ["Q5", "Reason support: SUPPORTED, PARTIAL, UNSUPPORTED, or UNSURE."],
    ["Q6", "Safety/workload: SAFE, CONCERN, UNSAFE, or UNSURE. A note is required for CONCERN/UNSAFE."],
    ["Privacy", "Keep reviewer_id pseudonymous (E##). Do not add names, emails, student IDs, or any personal data."],
    ["Submission", "Return the completed workbook without changing schema_version, reviewer_id, case_id, action_id, or randomized_order."],
  ];
  sheet.getRange(`A4:B${rows.length + 3}`).values = rows;
  sheet.getRange(`A4:A${rows.length + 3}`).format = {
    fill: palette.blue,
    font: { bold: true, color: palette.navy },
    verticalAlignment: "top",
  };
  sheet.getRange(`B4:B${rows.length + 3}`).format = {
    wrapText: true,
    verticalAlignment: "top",
  };
  sheet.getRange(`A4:B${rows.length + 3}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D8DEE6" },
    top: { style: "thin", color: palette.border },
    bottom: { style: "thin", color: palette.border },
    left: { style: "thin", color: palette.border },
    right: { style: "thin", color: palette.border },
  };
  sheet.getRange("A:A").format.columnWidth = 20;
  sheet.getRange("B:B").format.columnWidth = 95;
  sheet.getRange(`A4:B${rows.length + 3}`).format.rowHeight = 34;
}

function addCodebook(workbook) {
  const sheet = workbook.worksheets.add("Codebook");
  sheet.showGridLines = false;
  const rows = [
    ["Field", "Allowed values", "Interpretation"],
    ["q1_plan_score", "1, 2, 3, 4, 5", "Overall plan quality; ordinal."],
    ["q2_action_relevance", "APPROVE / PARTIAL / UNSURE / REJECT", "Score each proposed action separately."],
    ["q3_missing_action", "YES / NO", "Omission indicator; this is not action recall."],
    ["q4_escalation", "CORRECT / OVER_ESCALATED / UNDER_ESCALATED / UNSURE", "Human-review/escalation calibration."],
    ["q5_reason_support", "SUPPORTED / PARTIAL / UNSUPPORTED / UNSURE", "Whether displayed observed evidence supports the stated reasons."],
    ["q6_safety_workload", "SAFE / CONCERN / UNSAFE / UNSURE", "Safety and workload judgement."],
    ["reviewer_id", "E01, E02, ...", "Pseudonymous only; never put a real name here."],
  ];
  sheet.getRange(`A1:C${rows.length}`).values = rows;
  styleTable(sheet, `A1:C${rows.length}`, "A1:C1", {
    A: 25,
    B: 52,
    C: 70,
  });
  sheet.getRange(`A2:C${rows.length}`).format.wrapText = true;
  sheet.getRange(`A2:C${rows.length}`).format.rowHeight = 32;
}

async function buildReviewer(reviewerId, main = false) {
  const casesText = await fs.readFile(
    path.join(artifactDir, `expert_review_cases_${reviewerId}.csv`),
    "utf8",
  );
  const planText = await fs.readFile(
    path.join(artifactDir, `plan_review_template_${reviewerId}.csv`),
    "utf8",
  );
  const actionText = await fs.readFile(
    path.join(artifactDir, `action_review_template_${reviewerId}.csv`),
    "utf8",
  );
  const workbook = await Workbook.fromCSV(casesText, { sheetName: "Cases" });
  await workbook.fromCSV(planText, { sheetName: "Plan Review" });
  await workbook.fromCSV(actionText, { sheetName: "Action Review" });
  addInstructions(workbook);
  addCodebook(workbook);

  const cases = workbook.worksheets.getItem("Cases");
  styleTable(cases, "A1:V61", "A1:V1", {
    A: 24, B: 14, C: 15, D: 15, E: 18, F: 18, G: 24, H: 15, I: 21,
    J: 20, K: 20, L: 18, M: 21, N: 20, O: 28, P: 35, Q: 32, R: 72,
    S: 14, T: 14, U: 18, V: 62,
  });
  cases.freezePanes.freezeColumns(2);
  cases.getRange("A2:V61").format.wrapText = true;
  cases.getRange("A2:V61").format.rowHeight = 58;
  cases.getRange("G2:G61").conditionalFormats.add("containsText", {
    text: "ABSTAINED",
    format: { fill: palette.amber, font: { bold: true, color: "#9C5700" } },
  });

  const plan = workbook.worksheets.getItem("Plan Review");
  styleTable(plan, "A1:K61", "A1:K1", {
    A: 28, B: 12, C: 16, D: 14, E: 15, F: 18, G: 35, H: 22, I: 22, J: 40, K: 45,
  });
  plan.freezePanes.freezeColumns(4);
  plan.getRange("D2:D61").dataValidation = {
    rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
  };
  plan.getRange("E2:E61").dataValidation = {
    rule: { type: "list", values: ["YES", "NO"] },
  };
  plan.getRange("G2:G61").dataValidation = {
    rule: { type: "list", values: ["CORRECT", "OVER_ESCALATED", "UNDER_ESCALATED", "UNSURE"] },
  };
  plan.getRange("H2:H61").dataValidation = {
    rule: { type: "list", values: ["SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNSURE"] },
  };
  plan.getRange("I2:I61").dataValidation = {
    rule: { type: "list", values: ["SAFE", "CONCERN", "UNSAFE", "UNSURE"] },
  };
  plan.getRange("D2:K61").format.fill = "#FFFDF5";
  plan.getRange("F2:F61").format.wrapText = true;
  plan.getRange("J2:K61").format.wrapText = true;
  plan.getRange("D2:D61").conditionalFormats.add("cellIs", {
    operator: "lessThanOrEqual",
    formula: 2,
    format: { fill: palette.red, font: { bold: true, color: "#9C0006" } },
  });
  plan.getRange("D2:D61").conditionalFormats.add("cellIs", {
    operator: "greaterThanOrEqual",
    formula: 4,
    format: { fill: palette.green, font: { bold: true, color: "#375623" } },
  });

  const action = workbook.worksheets.getItem("Action Review");
  const actionUsed = action.getUsedRange(true);
  const actionRows = actionUsed.values.length;
  styleTable(action, `A1:G${actionRows}`, "A1:G1", {
    A: 28, B: 12, C: 16, D: 14, E: 28, F: 24, G: 55,
  });
  action.freezePanes.freezeColumns(5);
  if (actionRows > 1) {
    action.getRange(`F2:F${actionRows}`).dataValidation = {
      rule: { type: "list", values: ["APPROVE", "PARTIAL", "UNSURE", "REJECT"] },
    };
    action.getRange(`F2:G${actionRows}`).format.fill = "#FFFDF5";
    action.getRange(`G2:G${actionRows}`).format.wrapText = true;
  }

  const inspection = await workbook.inspect({
    kind: "workbook,sheet,table",
    maxChars: 8000,
    tableMaxRows: 4,
    tableMaxCols: 6,
    tableMaxCellChars: 80,
  });
  await fs.writeFile(
    path.join(outputDir, `inspection_${reviewerId}.json`),
    JSON.stringify(inspection, null, 2),
  );

  if (main) {
    for (const sheetName of ["Instructions", "Cases", "Plan Review", "Action Review", "Codebook"]) {
      const preview = await workbook.render({
        sheetName,
        autoCrop: "all",
        scale: 0.8,
        format: "png",
      });
      await fs.writeFile(
        path.join(renderDir, `${sheetName.replaceAll(" ", "_")}.png`),
        new Uint8Array(await preview.arrayBuffer()),
      );
    }
  }

  const blob = await SpreadsheetFile.exportXlsx(workbook);
  const reviewerOutput = path.join(outputDir, `expert_review_form_${reviewerId}.xlsx`);
  await blob.save(reviewerOutput);
  await blob.save(path.join(artifactDir, `expert_review_form_${reviewerId}.xlsx`));
  if (main) {
    await blob.save(path.join(outputDir, "expert_review_form.xlsx"));
    await blob.save(path.join(artifactDir, "expert_review_form.xlsx"));
  }
}

await buildReviewer("E01", true);
await buildReviewer("E02", false);
