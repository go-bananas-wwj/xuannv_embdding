import { readFileSync, writeFileSync } from "node:fs";

const input = new URL("china_v1_highres_open_models_report.html", import.meta.url);
const style = "<style>html{overflow-x:hidden}.analytics-top-bar{width:100%!important;max-width:100%!important}</style>";
const html = readFileSync(input, "utf8");

if (!html.includes(style)) {
  writeFileSync(input, html.replace("</head>", `${style}</head>`));
}
