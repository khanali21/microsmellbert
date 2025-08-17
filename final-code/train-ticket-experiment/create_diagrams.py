from __future__ import annotations
import argparse, os
from dataclasses import dataclass
from graphviz import Digraph

@dataclass(frozen=True)
class Theme:
    bg: str; edge: str; text: str; primary: str; secondary: str; accent: str; warn: str; muted: str

LIGHT = Theme("white","#6b7280","#111827","#2563eb","#059669","#7c3aed","#dc2626","#d1d5db")
DARK  = Theme("#0b1020","#9ca3af","#e5e7eb","#60a5fa","#34d399","#a78bfa","#f87171","#4b5563")
FONT = "Helvetica"

def new_graph(filename: str, theme: Theme, engine: str="dot", rankdir: str="LR",
              fileformat: str="png", dpi: int=300) -> Digraph:
    g = Digraph(filename=filename, format=fileformat, engine=engine)
    g.attr(rankdir=rankdir, pad="0.2", splines="spline", nodesep="0.35", ranksep="0.5")
    g.attr("graph", bgcolor=theme.bg, color=theme.edge, dpi=str(dpi))
    g.attr("node", shape="rect", style="rounded,filled", fillcolor="white",
           color=theme.muted, fontname=FONT, fontsize="11", fontcolor=theme.text, margin="0.08")
    g.attr("edge", color=theme.edge, arrowsize="0.7", penwidth="1.3")
    return g

def html_label(title: str, subtitle: str|None=None, mono: bool=False) -> str:
    title = title.replace("&","&amp;")
    subtitle = subtitle.replace("&","&amp;") if subtitle else None
    face = "monospace" if mono else FONT
    if subtitle:
        return f'''<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
<TR><TD><B><FONT POINT-SIZE="11" FACE="{FONT}">{title}</FONT></B></TD></TR>
<TR><TD><FONT POINT-SIZE="9" FACE="{face}" COLOR="#374151">{subtitle}</FONT></TD></TR>
</TABLE>>'''
    return f'''<<B><FONT POINT-SIZE="11" FACE="{FONT}">{title}</FONT></B>>'''

def generate_workflow_diagram(path: str, theme: Theme=LIGHT, fileformat: str="png", dpi: int=300) -> None:
    g = new_graph(path, theme, fileformat=fileformat, dpi=dpi)
    with g.subgraph(name="cluster_prep") as c:
        c.attr(label="Data & Preprocessing", color=theme.secondary, penwidth="1.8", style="rounded")
        c.node("repo", html_label("Train-Ticket Repo","Git submodules & services"))
        c.node("extract", html_label("Data Extraction","Java files, build graph"))
        c.node("pre", html_label("Preprocessing","strip comments, normalize ids"))
        c.node("label", html_label("Heuristic Labelling","Mega / CRUDy / Ambiguous"))
        c.edges([("repo","extract"),("extract","pre"),("pre","label")])
    with g.subgraph(name="cluster_feat") as c:
        c.attr(label="Feature Extraction", color=theme.primary, penwidth="1.8", style="rounded")
        c.node("ck", html_label("CK Metrics","LOC, CBO, RFC, WMC…", mono=True), fillcolor="#eff6ff")
        c.node("codebert", html_label("CodeBERT Embeddings","CLS (768-d)", mono=True), fillcolor="#eef2ff")
        c.edge("label","ck"); c.edge("label","codebert")
    with g.subgraph(name="cluster_ds") as c:
        c.attr(label="Dataset", color=theme.accent, penwidth="1.8", style="rounded")
        c.node("merge", html_label("Merge & Align","by class/service id"))
        c.node("split", html_label("Stratified Split","train / val / test 70/15/15"))
        c.edge("ck","merge"); c.edge("codebert","merge"); c.edge("merge","split")
    with g.subgraph(name="cluster_model") as c:
        c.attr(label="Modeling", color=theme.secondary, penwidth="1.8", style="rounded")
        c.node("train", html_label("Dual-Input NN","metrics MLP + embed MLP ➜ concat"))
        c.node("eval", html_label("Evaluation","acc / P / R / F1, PR curves"))
        c.node("infer", html_label("Inference","per-service predictions + scores"))
        c.edges([("split","train"),("train","eval"),("train","infer")])
    with g.subgraph(name="cluster_ci") as c:
        c.attr(label="DevOps Integration", color=theme.primary, penwidth="1.8", style="rounded")
        c.node("gha", html_label("CI Pipeline","GitHub Actions workflow"))
        c.node("annot", html_label("Human Feedback","dev review & corrections"))
        c.node("retrain", html_label("Model Retraining","periodic or threshold-based"))
        c.edge("infer","gha"); c.edge("gha","annot")
    g.edge("annot","retrain", style="dashed", color=theme.warn, penwidth="1.6")
    g.edge("retrain","train", style="dashed", color=theme.warn, penwidth="1.6",
           label="closed loop", fontsize="9", fontcolor=theme.warn)
    with g.subgraph(name="cluster_legend") as c:
        c.attr(label="Legend", color=theme.muted, style="dashed", penwidth="1.2")
        c.node("l1", html_label("Rounded box","processing step"))
        c.node("l2", html_label("Dashed red edge","feedback/retraining"))
        c.node("l3", html_label("Cluster","phase/stage"))
        c.edge("l1","l2", style="dashed", color=theme.warn); c.edge("l2","l3", style="solid")
    g.render(cleanup=True); print(f"Saved workflow diagram ➜ {path}")

def generate_model_diagram(path: str, theme: Theme=LIGHT, fileformat: str="png", dpi: int=300) -> None:
    g = new_graph(path, theme, fileformat=fileformat, dpi=dpi)
    with g.subgraph(name="cluster_inputs") as c:
        c.attr(label="Inputs", color=theme.muted, penwidth="1.6")
        c.node("x_metrics", html_label("Metrics Vector","k features (CK/CKJM)", mono=True), fillcolor="#eef2ff")
        c.node("x_embed", html_label("CodeBERT CLS","768-d", mono=True), fillcolor="#f0fdf4")
    with g.subgraph(name="cluster_metrics") as c:
        c.attr(label="Metrics Branch", color=theme.primary, penwidth="1.8", style="rounded")
        c.node("m_dense1", html_label("Dense 128","ReLU + Dropout 0.2", mono=True))
        c.node("m_dense2", html_label("Dense 64","ReLU", mono=True))
        c.edge("x_metrics","m_dense1"); c.edge("m_dense1","m_dense2")
    with g.subgraph(name="cluster_embed") as c:
        c.attr(label="Embedding Branch", color=theme.secondary, penwidth="1.8", style="rounded")
        c.node("e_dense1", html_label("Dense 256","ReLU + Dropout 0.2", mono=True))
        c.node("e_dense2", html_label("Dense 128","ReLU", mono=True))
        c.edge("x_embed","e_dense1"); c.edge("e_dense1","e_dense2")
    g.node("concat", html_label("Concatenate","axis=-1"), shape="ellipse", style="filled", fillcolor="#fff7ed")
    g.edge("m_dense2","concat"); g.edge("e_dense2","concat")
    with g.subgraph(name="cluster_head") as c:
        c.attr(label="Classification Head", color=theme.accent, penwidth="1.8", style="rounded")
        c.node("h_dense1", html_label("Dense 128","BatchNorm + ReLU + Dropout 0.3", mono=True))
        c.node("h_out", html_label("Dense 3","Sigmoid (one-vs-rest)", mono=True))
        c.edge("concat","h_dense1"); c.edge("h_dense1","h_out")
    g.node("y", html_label("Outputs","P(Mega), P(CRUDy), P(Ambiguous)"))
    g.attr("edge", fontname=FONT, fontsize="9")
    g.edge("h_out","y", xlabel="3×1")
    g.edge("x_metrics","m_dense1", xlabel="k×1")
    g.edge("x_embed","e_dense1", xlabel="768×1")
    g.edge("concat","h_dense1", xlabel="(64+128)×1")
    g.render(cleanup=True); print(f"Saved model diagram ➜ {path}")

def main():
    parser = argparse.ArgumentParser(description="Generate thesis diagrams.")
    parser.add_argument("--workflow", default=None, help="Path to save workflow diagram (e.g., figures/workflow.png)")
    parser.add_argument("--model", default=None, help="Path to save model architecture diagram (e.g., figures/model.png)")
    parser.add_argument("--all", action="store_true", help="Generate both diagrams with defaults (figures/).")
    parser.add_argument("--theme", choices=["light","dark"], default="light", help="Color theme.")
    parser.add_argument("--format", choices=["png","svg"], default="png", help="Output format.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI for PNG/SVG export.")
    args = parser.parse_args()

    theme = LIGHT if args.theme == "light" else DARK
    os.makedirs("figures", exist_ok=True)

    if args.all or args.workflow:
        path = args.workflow or os.path.join("figures","workflow.png")
        generate_workflow_diagram(path, theme, fileformat=args.format, dpi=args.dpi)

    if args.all or args.model:
        path = args.model or os.path.join("figures","model.png")
        generate_model_diagram(path, theme, fileformat=args.format, dpi=args.dpi)

if __name__ == "__main__":
    main()
