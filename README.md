# Generative-AI-Project

As part of the project, we aim to evaluate and improve these models systematically by establishing the baseline performance using basic prompts, enhancing generation quality through advanced prompting techniques, and fine-tuning the models to further improve their performance. REMARK: The course project draws inspiration from a recent paper. https://openreview.net/pdf?id=JRMSC08gSF

The first part of the project focuses on evaluating the baseline models and exploring prompt engineering techniques for program repair and hint generation. You will use the provided datasets and scripts to analyze the performance of models such as **GPT-4o-mini** and **Phi-3-mini**. We will explore advanced prompting techniques to improve the generation quality.

## Provided Datasets and Scripts

To use the INTROPYNUS dataset [1, 2] for evaluation, consisting of tuples with a programming
task $T$ , a buggy program $P_b$, repaired program $P_r$, and test suite $Ω_T$. There are 5 tasks, each with 5 buggy programs.

The project_part1_datasets/ containing two directories, namely problems/ and evaluation_data/.

The problems/ directory contains JSON files, each representing an individual programming problem. Each JSON file includes a textual problem description, the buggy program, the correct repaired program, and a set of test cases with inputs and expected outputs. 

The evaluation_data/ directory provides the detailed evaluation setup for each problem. It is organized into problem-based folders, where each folder corresponds to a specific programming problem (e.g., 1_sequential-search). Inside each problem folder, subfolders (e.g., prog_1) contain the buggy and repaired Python files (buggy.py and fixed.py) for the individual problems.

Remember to change the «OPENAI_API_KEY_FILE» placeholder in project_part1_repair.py and project_part1_hint.py with its actual name.

[1] Nachiket Kotalwar, Alkis Gotovos, and Adish Singla. Hints-In-Browser: Benchmarking Lan-
guage Models for Programming Feedback Generation. In NeurIPS (Datasets and Benchmarks
Track), 2024. Paper link: https://openreview.net/pdf?id=JRMSC08gSF

[2] Yang Hu, Umair Z. Ahmed, Sergey Mechtaev, Ben Leong, and Abhik Roychoudhury. Re-
Factoring Based Program Repair Applied to Programming Assignments. In ASE, 2019. Paper
link: https://mechtaev.com/files/ase19.pdf.
