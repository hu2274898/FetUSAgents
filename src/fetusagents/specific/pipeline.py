"""5-Voter / Expert / Checker / Report-Generator pipeline.

This is the top of the specific VQA stack. It loads ``TASK_SPECS``,
builds the AutoGen agents (analyst, 5 voters, checker, report
generator), runs ``solve_one_sample`` per sample, computes metrics, and
optionally streams everything to disk.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Schemas / data classes
from .types import (
    AgentVote,
    KnowledgeStore,
    SampleResult,
    SAVE_EVERY,
    TaskSpec,
    ToolDecision,
    VQASample,
    load_checkpoint,
    save_checkpoint,
)

# Knowledge / option mappings
from .knowledge import (
    BRAIN_OPTION_TO_TEXT,
    PLANE_OPTION_TO_TEXT,
    TRIMESTER_OPTION_TO_TEXT,
    YESNO_OPTION_TO_TEXT,
    collect_knowledge,
    extract_reason_from_voter,
)

# Parsers / helpers
from .parsers import (
    count_votes,
    get_consensus_answer,
    load_vqa_dataset,
    make_mm_message,
    normalize_space,
    parse_abcd_answer,
    parse_brain_subplane_answer,
    parse_plane_answer,
    parse_trimester_multi_answer,
    parse_yesno_answer,
)

# Tool agents (one per task) — registered as ToolAgent instances so
# they expose a uniform ``name`` / ``description`` / ``tools`` surface
# alongside the general-workflow experts.
from .experts import SPECIFIC_EXPERTS


# OpenAI client (overrides the simpler one in fetus_core.llm with a
# vision-/json-/structured-aware client)
def build_model_client() -> OpenAIChatCompletionClient:
    model_name = os.environ.get("OPENAI_MODEL", "gpt-5.1")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
        },
    )


# Dataset roots
# The original FetUS-VQA layout uses two parallel trees:
#   $FETUSAGENTS_VQA_DIR   — VQA JSONs   (e.g. .../Agent_VQA/Fetal_plane/vqa_plane_multi.json)
#   $FETUSAGENTS_DATA_DIR  — image roots (e.g. .../FU-LoRA-main/data/classification/african_test)
# Both are resolved at TASK_SPECS-construction time. CLI overrides
# (--vqa_json / --image_dir) still take precedence per-call.
_VQA_DIR = os.environ.get("FETUSAGENTS_VQA_DIR", "./Agent_VQA")
_DATA_DIR = os.environ.get("FETUSAGENTS_DATA_DIR", "./datasets")


def _vqa(rel: str) -> str:
    return os.path.join(_VQA_DIR, rel)


def _data(rel: str) -> str:
    return os.path.join(_DATA_DIR, rel)


TASK_SPECS: Dict[str, TaskSpec] = {
    "plane_classification": TaskSpec(
        name="plane_classification",
        vqa_json=_vqa("Fetal_plane/vqa_plane_multi.json"),
        default_image_dir=_data("FU-LoRA-main/data/classification/african_test"),
        option_to_text=PLANE_OPTION_TO_TEXT,
        allowed_letters=("A", "B", "C", "D"),
        answer_parser=parse_plane_answer,
        expert=SPECIFIC_EXPERTS["plane_classification"],
    ),
    "brain_subplane": TaskSpec(
        name="brain_subplane",
        vqa_json=_vqa("Brain_class/vqa_brain_subplane_multi.json"),
        default_image_dir=_data("dataset_agent/brain_subplane_test/3brain_agent"),
        option_to_text=BRAIN_OPTION_TO_TEXT,
        allowed_letters=("A", "B", "C"),
        answer_parser=parse_brain_subplane_answer,
        expert=SPECIFIC_EXPERTS["brain_subplane"],
    ),
    "brain_subplane_binary": TaskSpec(
        name="brain_subplane_binary",
        vqa_json=_vqa("Brain_class/vqa_brain_subplane_binary.json"),
        default_image_dir=_data("dataset_agent/brain_subplane_test/3brain_agent"),
        option_to_text=YESNO_OPTION_TO_TEXT,
        allowed_letters=("A", "B"),
        answer_parser=parse_yesno_answer,
        expert=SPECIFIC_EXPERTS["brain_subplane_binary"],
    ),
    "ga_trimester_binary": TaskSpec(
    name="ga_trimester_binary",
    vqa_json=_vqa("GA_estimation/vqa_trimester_binary.json"),
    default_image_dir=_data("CSM-for-fetal-HC-measurement-main/data/HC18_dataset_real_hc18/training_set"),
    option_to_text=YESNO_OPTION_TO_TEXT,
    allowed_letters=("A", "B"),
    answer_parser=parse_yesno_answer,
    expert=SPECIFIC_EXPERTS["ga_trimester_binary"],
),
    "hc_estimation_pixel": TaskSpec(
        name="hc_estimation_pixel",
        vqa_json=_vqa("HC/vqa_hc_estimation.json"),
        default_image_dir=_data("CSM-for-fetal-HC-measurement-main/data/validation_out/images"),
        option_to_text={
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
        },
        allowed_letters=("A", "B", "C", "D"),
        answer_parser=parse_abcd_answer,
        expert=SPECIFIC_EXPERTS["hc_estimation_pixel"],
    ),
    "ga_trimester_multi": TaskSpec(
    name="ga_trimester_multi",
    vqa_json=_vqa("GA_estimation/vqa_trimester_multi.json"),
    default_image_dir=_data("CSM-for-fetal-HC-measurement-main/data/HC18_dataset_real_hc18/training_set"),
    option_to_text=TRIMESTER_OPTION_TO_TEXT,
    allowed_letters=("A", "B", "C"),
    answer_parser=parse_trimester_multi_answer,
    expert=SPECIFIC_EXPERTS["ga_trimester_multi"],
),
    "plane_binary": TaskSpec(
        name="plane_binary",
        vqa_json=_vqa("Fetal_plane/vqa_plane_binary.json"),
        default_image_dir=_data("FU-LoRA-main/data/classification/african_test"),
        option_to_text=YESNO_OPTION_TO_TEXT,
        allowed_letters=("A", "B"),
        answer_parser=parse_yesno_answer,
        expert=SPECIFIC_EXPERTS["plane_binary"],
    ),
    "aop_binary": TaskSpec(
        name="aop_binary",
        vqa_json=_vqa("Aop/vqa_binary_aop_dataset_answer.json"),
        default_image_dir=_data("USFM-master/datasets/Seg/toy_dataset/test_set/image_data1_2023"),
        option_to_text={
            "A": "AoP >= 120°, spontaneous vaginal delivery is indicated",
            "B": "AoP < 120°, instrumental delivery or cesarean may be necessary",
        },
        allowed_letters=("A", "B"),
        answer_parser=parse_yesno_answer,
        expert=SPECIFIC_EXPERTS["aop_binary"],
    ),
    "ac_estimation_pixel": TaskSpec(
        name="ac_estimation_pixel",
        vqa_json=_vqa("AC/vqa_ac_estimation.json"),
        default_image_dir=_data("dataset_agent/AC_new/img_with_gt"),
        option_to_text={
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
        },
        allowed_letters=("A", "B", "C", "D"),
        answer_parser=parse_abcd_answer,
        expert=SPECIFIC_EXPERTS["ac_estimation_pixel"],
    ),
    "stomach_volume_estimation": TaskSpec(
        name="stomach_volume_estimation",
        vqa_json=_vqa("Stomach/vqa_stomach_volume_estimation.json"),
        default_image_dir=_data("dataset_agent/abdomen_seg_2/images_select_png"),
        option_to_text={
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
        },
        allowed_letters=("A", "B", "C", "D"),
        answer_parser=parse_abcd_answer,
        expert=SPECIFIC_EXPERTS["stomach_volume_estimation"],
    ),
}

# build agents
def build_agents(model_client: OpenAIChatCompletionClient, task_spec: TaskSpec) -> Dict[str, AssistantAgent]:
    allowed = "|".join(task_spec.allowed_letters)

    allocator = AssistantAgent(
        name="task_allocator",
        model_client=model_client,
        system_message=(
            "You are the Task Allocation Agent for fetal ultrasound VQA.\n"
            "Your output must contain exactly two lines:\n"
            "Task: <task_name>\n"
            "Reason: <short explanation>\n"
        ),
    )

    analyst = AssistantAgent(
        name="question_analyst",
        model_client=model_client,
        system_message=(
            "You are the Question Analysis Agent.\n"
            "Explain what the question asks, what visual evidence matters, and which options are easy to confuse.\n"
            "Do not provide the final answer.\n"
            "Output 3 to 6 sentences only."
        ),
    )


    voter_roles = {
        "voter_1": ( 
            "You are Voter 1, a senior fetal ultrasound anatomy specialist.\n" 
            "Your role is to make anatomy-first judgments based on visible fetal structures and reliable sonographic landmarks.\n"
            "Focus on identifying the dominant anatomical region, standard imaging plane, and key structures such as the skull, brain, abdomen, femur, thorax, stomach, heart, pelvis, or other clearly visible landmarks.\n" 
            "When classification or measurement is required, reason from concrete anatomical evidence rather than global appearance alone.\n" "For uncertain cases, explicitly rely on the most discriminative visible landmarks and avoid over-interpreting weak or ambiguous image cues." ),
        "voter_2": (
            "You are Voter 2, a fetal ultrasound visual pattern specialist.\n"
            "Your role is to make image-driven judgments based on visible appearance, geometry, contrast, contour, texture, symmetry, and spatial layout.\n"
            "Focus on the dominant visual pattern of the image, including whether structures appear circular, elliptical, elongated, cavity-like, bone-dominant, or arranged like a standard sonographic plane.\n"
            "When classification or measurement is required, reason from concrete visual evidence such as object shape, boundary extent, relative position, brightness distribution, and scale within the frame.\n"
            "For uncertain cases, rely on the clearest visible pattern cues and avoid making conclusions from weak, noisy, or ambiguous image features."
        ),
        "voter_3": ( 
            "You are Voter 3, an elimination-based diagnostic reasoner.\n" 
            "Your role is to compare candidate answers systematically and identify the best-supported option by ruling out weaker alternatives.\n"
            "Evaluate which choices are inconsistent with the visible evidence, which key landmarks or patterns are missing, and which remaining option is most strongly supported.\n"
            "When classification or measurement is required, first exclude clearly implausible candidates based on anatomy, visual pattern, geometry, scale, or contextual consistency.\n" 
            "For uncertain cases, avoid premature commitment; explain which alternatives are weakly supported and select the option with the strongest remaining evidence." ),
        "voter_4": (
            "You are Voter 4, a conservative and uncertainty-aware reviewer.\n"
            "Your role is to assess the reliability of the visible evidence and prevent overconfident conclusions.\n"
            "Pay attention to ambiguity caused by incomplete views, weak contrast, artifacts, atypical anatomy, partial landmarks, or overlap between similar categories.\n"
            "When classification or measurement is required, prefer the option that remains most defensible under uncertainty rather than one that is only superficially plausible.\n"
            "For uncertain cases, explicitly acknowledge limited evidence, avoid extreme judgments, and choose the most robustly supported answer."
        ),
        "voter_5": (
            "You are Voter 5, an integrated final-judgment expert.\n"
            "Your role is to synthesize anatomical evidence, visual patterns, option comparison, and uncertainty calibration into a balanced conclusion.\n"
            "Evaluate the image as a whole rather than relying on any single isolated cue, and identify the answer that is most consistent with the combined evidence.\n"
            "When classification or measurement is required, integrate visible landmarks, global appearance, geometry, scale, and contextual plausibility to make the most defensible decision.\n"
            "For uncertain cases, weigh competing evidence carefully and provide a decisive final choice only when it is sufficiently supported."
        ),
    }
    agents: Dict[str, AssistantAgent] = {
        "allocator": allocator,
        "analyst": analyst,
    }

    for name, role in voter_roles.items():
        agents[name] = AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=(
                role + "\n"
                "You must output only in the following format:\n"
                "Reason: <1-3 sentence justification>\n"
                f"Final Answer: <{allowed}>\n"
            ),
        )

    agents["checker"] = AssistantAgent(
        name="final_decision_checker",
        model_client=model_client,
        system_message=(
            "You are the final decision Agent for fetal ultrasound Visual Question Answering.\n\n"
            "You will receive ALL available evidence for a question:\n"
            "  (a) Question analysis from an analyst agent\n"
            "  (b) Votes and detailed reasoning from 5 independent voter agents\n"
            "  (c) Results from specialized, validated medical-imaging TOOL algorithms\n\n"
            "Your sole job is to output the single best answer.\n\n"
            "Decision Guidance\n"
            "the specialized tool is designed for this specific task and is usually considered a reliable source of evidence. "
            "however, the voter results should also be carefully considered. "
            "if a majority of voters give the same answer and their reasoning is highly persuasive, "
            "with concrete, specific, and image-grounded evidence, you may follow the voter consensus even when it differs from the tool answer. "
            "vague or generic disagreement is not sufficient. "
            "if the tool answer is none, rely on the voter results and their supporting evidence.\n\n"
            "Output Format (strictly):\n"
            f"Final Answer: <one of {allowed}>\n"
            "Reason: <1-3 sentences: state whether you followed the tool or\n"
            "the voters, and briefly explain why>\n"
        ),
    )
    agents["report_generator"] = AssistantAgent(
        name="report_generator",
        model_client=model_client,
        system_message=(
            "You are an experienced prenatal ultrasound specialist with many years "
            "of clinical practice in fetal ultrasound."
            "You will receive all evidence collected during an automated fetal ultrasound image analysis session:\n"
            "1. Expert voter reasoning (5 independent specialists)\n"
            "2. Quantitative tool measurements (segmentation, biometry, classification)\n"
            "3. Retrieved medical knowledge from authoritative literature (RAG)\n"
            "4. The analyst's question decomposition\n"
            "5. The final consensus answer\n\n"
            "Your task is to synthesize the evidence into a concise, professionally written clinical analysis report.\n\n"
            "Output Format\n"
            "Findings: <Describe the observable anatomical structures, visual landmarks, and image features.>\n"
            "Impressions: <State only the final clinical interpretation, classification result, or measurement outcome concisely.>\n"
            "Note: <Summarize supporting evidence in this order: AI tool results first, expert agreement second, RAG literature support third.>\n\n"
            "evidence integration policy\n"
            "Prioritize the final consensus answer and tool-supported evidence.\n"
            "When expert readers and automated tools agree, integrate expert observations naturally and state concordance briefly.\n"
            "When expert readers disagree with tools, include only shared or non-controversial visual observations in Findings, and summarize disagreement briefly in Note.\n"
            "Avoid abrupt contradiction between Findings and Impressions.\n"
            "Do not force every evidence source into the report if doing so makes the report awkward.\n\n"
            "Writing Requirements\n"
            "1. Total length: 30-60 words.\n"
            "2. Professional medical language; concise, smooth, and coherent.\n"
            "3. Do not fabricate measurements, structures, or certainty.\n"
            "4. Integrate RAG knowledge naturally; do not copy or list retrieved chunks.\n"
            "5. If segmentation-based measurement was clearly performed, mention it briefly and name the tool and segmented structure.\n"
            "Style examples\n"
            "If tools and experts agree, write smoothly like:\n"
            "'Note: Automated classification by [tool] supported [result]; expert review was concordant, and RAG references supported this interpretation.'\n\n"
            "If tools and experts disagree, write smoothly like:\n"
            "'Note: Automated classification by [tool] supported [result]; expert visual interpretations were mixed, so the final impression follows the tool-supported result. RAG references provided contextual support.'\n\n"
            "Return only the three required sections: Findings, Impressions, and Note."
        ),
    )
    return agents


# run agent 
async def run_text_agent(agent: AssistantAgent, prompt: Any, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            result = await agent.run(task=prompt)
            if getattr(result, "messages", None):
                for m in reversed(result.messages):
                    content = getattr(m, "content", None)
                    if isinstance(content, str) and content.strip():
                        return content
            return ""

        except Exception as e:
            error_name = type(e).__name__
            if attempt < max_retries:
                wait = 5 * (2 ** (attempt - 1))  # 5, 10, 20, 40, 80
                print(f"    [RETRY {attempt}/{max_retries}] {error_name}: {e}")
                print(f"    Waiting {wait}s ...")
                await asyncio.sleep(wait)
            else:
                print(f"    [FAILED after {max_retries} retries] {error_name}: {e}")
                raise


async def solve_one_sample(sample: VQASample, agents: Dict[str, AssistantAgent]) -> SampleResult:
    if not os.path.exists(sample.image_path):
        raise FileNotFoundError(f"can not find image: {sample.image_path}")

    task_spec = TASK_SPECS[sample.task_name]
    option_to_text = task_spec.option_to_text

    allocator_text = f"Task: {sample.task_name}\nReason: task is specified by CLI."
    task_type = sample.task_name

    analysis_text_prompt = (
        f"Question: {sample.question}\n"
        f"Options: {json.dumps(sample.options, ensure_ascii=False)}\n"
        f"Task type: {task_type}\n"
        "Please analyze what this question is asking and what visual evidence should be examined. "
        "Do not provide the final answer."
    )
    analysis_prompt = make_mm_message(analysis_text_prompt, sample.image_path)
    analysis_text = await run_text_agent(agents["analyst"], analysis_prompt)

    voter_names = ["voter_1", "voter_2", "voter_3", "voter_4", "voter_5"]
    voter_text_prompt = (
        f"Task type: {task_type}\n"
        f"Question: {sample.question}\n"
        f"Options: {json.dumps(sample.options, ensure_ascii=False)}\n"
        f"Question analysis: {analysis_text}\n"
        "Now vote independently based on the image."
    )

    async def run_voter(voter_name: str) -> AgentVote:
        voter_prompt = make_mm_message(voter_text_prompt, sample.image_path)
        raw = await run_text_agent(agents[voter_name], voter_prompt)
        letter = task_spec.answer_parser(raw)
        return AgentVote(
            agent_name=voter_name,
            answer_letter=letter,
            answer_text=option_to_text.get(letter) if letter else None,
            raw_output=raw,
        )
    async def run_tool_async() -> ToolDecision:
        """Put the synchronous tool runner into a thread pool so it can run in parallel with the voter API calls."""
        if task_spec.expert is not None and task_spec.expert.runner is not None:
            try:
                return await asyncio.to_thread(task_spec.expert.runner, sample)
            except Exception as e:
                print(f"  [TOOL ERROR] {type(e).__name__}: {e}")
        return ToolDecision(
            used_tool=False,
            tool_name=None,
            tool_answer_letter=None,
            tool_answer_text=None,
            tool_detail={"note": "tool unavailable or failed"},
        )
    voter_results, tool_decision = await asyncio.gather(
        asyncio.gather(*[run_voter(name) for name in voter_names]),
        run_tool_async(),
    )
    votes: List[AgentVote] = list(voter_results)
    vote_count = count_votes(votes)
    consensus_answer = get_consensus_answer(vote_count, threshold=4)

    voter_section_lines: List[str] = []
    for v in votes:
        ans_display = (
            f"{v.answer_letter} ({option_to_text.get(v.answer_letter, '?')})"
            if v.answer_letter else "NONE (parse failed)"
        )
        raw_trimmed = v.raw_output[:600] if v.raw_output else "(empty)"
        voter_section_lines.append(
            f"[{v.agent_name}]  Answer: {ans_display}\n"
            f"  Reasoning: {raw_trimmed}"
        )
    voter_section = "\n\n".join(voter_section_lines)
    vote_tally_str = ", ".join(
        f"{k}({option_to_text.get(k, '?')})={v}" for k, v in vote_count.items()
    )
    consensus_display = (
        f"{consensus_answer} ({option_to_text.get(consensus_answer, '?')})"
        if consensus_answer else "NO CONSENSUS"
    )
    if tool_decision.used_tool and tool_decision.tool_answer_letter is not None:
        tool_section = (
            f"Tool name   : {tool_decision.tool_name}\n"
            f"Tool answer : {tool_decision.tool_answer_letter} "
            f"({tool_decision.tool_answer_text})\n"
            f"Tool details: {json.dumps(tool_decision.tool_detail, ensure_ascii=False, default=str)}"
        )
    elif tool_decision.used_tool:
        tool_section = (
            f"Tool name   : {tool_decision.tool_name}\n"
            f"Tool answer : NONE  (tool ran but could not determine an answer)\n"
            f"Tool details: {json.dumps(tool_decision.tool_detail, ensure_ascii=False, default=str)}"
        )
    else:
        tool_section = "No tool was available or the tool failed to execute."
    checker_prompt_text = (
        "you are the final decision maker. review all available evidence.\n"
        f"question\n{sample.question}\n\n"
        f"options\n" + "\n".join(sample.options) + "\n\n"
        f"question analysis by analyst\n{analysis_text}\n\n"
        "voter results from 5 independent voters\n"
        f"{voter_section}\n\n"
        f"vote tally\n{vote_tally_str}\n"
        f"voter consensus: {consensus_display}\n\n"
        "specialized tool result\n"
        f"{tool_section}\n\n"
        "decision guidance\n"
        "the specialized tool is designed for this specific task and is usually considered a reliable source of evidence. "
        "however, the voter results should also be carefully considered. "
        "if a majority of voters give the same answer and their reasoning is highly persuasive, "
        "with concrete, specific, and image-grounded evidence, you may follow the voter consensus even when it differs from the tool answer. "
        "vague or generic disagreement is not sufficient. "
        "if the tool answer is none, rely on the voter results and their supporting evidence.\n\n"
        "now give your final decision."
    )
    checker_prompt = make_mm_message(checker_prompt_text, sample.image_path)
    checker_text = await run_text_agent(agents["checker"], checker_prompt)
    final_answer = task_spec.answer_parser(checker_text)
    route = "checker_decision"
    if final_answer is None:
        if tool_decision.tool_answer_letter is not None:
            final_answer = tool_decision.tool_answer_letter
            route = "checker_parse_failed→tool_fallback"
        elif consensus_answer is not None:
            final_answer = consensus_answer
            route = "checker_parse_failed→consensus_fallback"
        elif vote_count:
            final_answer = next(iter(vote_count.keys()))
            route = "checker_parse_failed→majority_fallback"
        else:
            route = "all_failed"

    knowledge = collect_knowledge(
        sample=sample,
        votes=votes,
        tool_decision=tool_decision,
        analysis_text=analysis_text,
        final_answer=final_answer,
        option_to_text=option_to_text,
    )

    report_text = ""
    try:
        voter_reason_lines = []
        for vr in knowledge.voter_reasons:
            voter_reason_lines.append(
                f"[{vr['agent_name']}]  Answer: {vr['answer_letter']} ({vr['answer_text']})\n"
                f"  Reasoning: {vr['reason']}"
            )
        voter_section_for_report = "\n\n".join(voter_reason_lines)
        ts = knowledge.tool_summary
        if ts.get("used_tool"):
            tool_lines = [f"Tool: {ts.get('tool_name', 'N/A')}"]
            tool_lines.append(
                f"Tool Answer: {ts.get('tool_answer_letter', 'N/A')} "
                f"({ts.get('tool_answer_text', 'N/A')})"
            )
            for display_key, detail_key in [
                ("Predicted pixel value", "pred_pixel"),
                ("Estimated total weeks", "recommended_total_weeks"),
                ("Estimated AoP (degrees)", "recommended_aop_deg"),
                ("Predicted trimester", "predicted_trimester"),
                ("Predicted plane label", "predicted_label"),
                ("Target label", "target_label"),
                ("AoP threshold (degrees)", "threshold_deg"),
                ("Option value map", "option_map"),
            ]:
                if detail_key in ts and ts[detail_key] is not None:
                    tool_lines.append(f"{display_key}: {ts[detail_key]}")
            tool_section_for_report = "\n".join(tool_lines)
        else:
            tool_section_for_report = "No quantitative tool data available."
        if knowledge.rag_knowledge:
            rag_section = "\n---\n".join(
                f"[Knowledge #{i+1}]\n{chunk}"
                for i, chunk in enumerate(knowledge.rag_knowledge[:5])
            )
        else:
            rag_section = "No relevant medical literature was retrieved."
        report_prompt_text = (
            "generate a structured fetal ultrasound analysis report using all "
            "the evidence below.\n\n"
            f"study information\n"
            f"image id: {sample.image_id}\n"
            f"task type: {sample.task_name}\n"
            f"question: {sample.question}\n"
            f"options: {json.dumps(sample.options, ensure_ascii=False)}\n\n"
            f"question analysis\n"
            f"{knowledge.analysis_text}\n\n"
            f"expert voter reasoning from knowledge module\n"
            f"{voter_section_for_report}\n\n"
            f"quantitative tool measurements\n"
            f"{tool_section_for_report}\n\n"
            f"medical knowledge from rag\n"
            f"{rag_section}\n\n"
            f"final answer\n"
            f"{final_answer} ({option_to_text.get(final_answer, 'N/A')})\n\n"
            "now write the report."
        )
        report_prompt = make_mm_message(report_prompt_text, sample.image_path)
        report_text = await run_text_agent(agents["report_generator"], report_prompt)
        print(f"  [REPORT] Generated ({len(report_text)} chars)")
    except Exception as e:
        print(f"  [REPORT ERROR] {type(e).__name__}: {e}")
        report_text = f"(Report generation failed: {e})"

    correct = (final_answer == sample.answer)
    return SampleResult(
        image_id=sample.image_id,
        gt_answer=sample.answer,
        task_type=task_type,
        allocator_text=allocator_text,
        analysis_text=analysis_text,
        votes=votes,
        vote_count=vote_count,
        consensus_answer=consensus_answer,
        final_answer=final_answer,
        route=route,
        checker_text=checker_text,
        tool_decision=tool_decision,
        correct=correct,
        report=report_text,
        knowledge_store=knowledge,
    )


