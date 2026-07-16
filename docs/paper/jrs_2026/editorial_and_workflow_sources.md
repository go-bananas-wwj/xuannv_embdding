# Editorial and Scientific-Writing Sources

## Target Journal

- [Journal of Remote Sensing: Guidelines for Authors](https://spj.science.org/page/remotesensing/for-authors)
- [Special Issue: Foundation Models based Multimodal Earth Observation Data Fusion and Applications](https://spj.science.org/page/remotesensing/si/foundation-models-multimodal-earth-observation-data-fusion)

The framework applies the following checked requirements:

- Research Article structure includes Title, Abstract, Introduction, Materials and Methods, Results, Discussion, declarations, references, figures/tables, and supplementary materials.
- The abstract is capped at 250 words.
- The working manuscript target is below 15,000 words.
- The main-text budget is no more than 10 figures and tables combined.
- The first Methods subsection is `Experimental and Technical Design` and is paired with a workflow diagram.
- At revision, each figure is delivered as a separate editable file. Diagrams use PDF/EPS; raster remote-sensing panels use an accepted raster/PDF format at a minimum of 300 dpi. PowerPoint/Word figures and converted PowerPoint/Word artwork are not accepted.
- Data, code, material availability, prior publication, and image permissions are treated as submission requirements rather than afterthoughts.

## Research-Writing Skills Consulted

- [NousResearch Hermes: Research Paper Writing Skill](https://github.com/nousresearch/hermes-agent/blob/main/skills/research/research-paper-writing/SKILL.md)
- [SciWrite: Scientific Manuscript Review Skill](https://github.com/labarba/sciwrite)
- [ByteDance DeerFlow: Academic Paper Review Skill](https://github.com/bytedance/deer-flow/blob/main/skills/public/academic-paper-review/SKILL.md)

Practices adopted from these resources:

1. Lock a one-sentence contribution before drafting sections.
2. Use a figure-first plan in which every main figure answers one research question.
3. Draft the abstract as background, gap, method, result, and significance; insert results only after verification.
4. Separate factual drafting from a second prose-refinement pass.
5. Review methodology, novelty, significance, literature positioning, limitations, active voice, terminology consistency, numerical integrity, citation integrity, and reproducibility in separate passes.
6. Keep an explicit claim-evidence matrix so unsupported claims cannot migrate into the abstract or conclusion.

## Project Sources of Truth

- `docs/agent_memory/haidian_embedding_project_memory.md`
- `docs/paper/xuannv_paper_experiment_protocol_20260715_zh.md`
- `configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml`
- `src/xuannv_embedding/models/`
- `src/xuannv_embedding/training/losses.py`

The implementation and immutable experiment artifacts override historical report prose whenever they disagree.
