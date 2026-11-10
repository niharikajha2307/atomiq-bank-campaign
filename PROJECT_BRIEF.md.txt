Atomiq Hackathon: Claude Handoff Pack
and Antigravity Guide
Hybrid Quantum-Classical Term Deposit Campaign Pipeline. Prepared for a beginner who
will build everything inside Google Antigravity.
What this document is
This is a copy-paste kit. It has three things:
1. A step-by-step guide to set up Google Antigravity and use Claude inside it.
2. A project brief you save as a file, so Claude always knows the full problem.
3. Ready-made prompts for each stage, so you never have to guess what to ask.
You do not need prior coding or quantum experience. Claude writes and runs the code.
Your job is to give it the right information, run it one stage at a time, and check each result.
Part 1: Set up Google Antigravity
Antigravity is Google's AI coding tool. It looks like a code editor with an AI agent built in.
The agent can create files, run commands, and fix its own errors. Menu names can change
between versions, so if something below looks slightly different on your screen, look for
the closest match.
1. Install Antigravity. Search for Google Antigravity and download it from Google's
official page only. Install it and sign in with your Google account.
2. Install Python. Download Python 3.10 or newer from python.org. On Windows, tick the
box that says Add Python to PATH during installation.
3. Make a project folder. Create a folder called atomiq-bank-campaign on your
computer.
4. Open the folder in Antigravity. Use File, then Open Folder, and choose atomiq-bankcampaign . Everything the agent creates will live here.
5. Choose the model. In the agent panel there is a model selector. If a Claude model is
listed, choose the strongest one available. If no Claude model appears for your
account, use the best model you can see. The same prompts below work with any of
them.
6. Put your files in the folder. See Part 2.
7. Paste the master prompt from Part 4 into the agent panel and send it.
8. Work one stage at a time. Use the stage prompts in Part 5. After each stage, read the
agent's summary, check the output files, and only then move on.
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 1 of 11
Habits that save you from trouble
IfAntigravity offers a plan or review step before the agent starts, read the plan. If it
touches the test labels or skips a rule, tell it to fix that first.
Ask the agent to explain anything you do not understand in plain words. This also
helps you write the presentation.
Save your work often. Ask the agent to write a short progress note in PROGRESS.md at
the end of every session, so a new chat can pick up where you left off.
If something breaks, paste the full error message to the agent. Do not retype it from
memory.
Part 2: What to put in your project folder
Create this layout. The agent can create the empty folders for you.
atomiq-bank-campaign/
inputs/
PROBLEM_STATEMENT_THEATOMIQHACKTHON.pdf
bank_campaign_participant_release.xlsx
PROJECT_BRIEF.md
PROGRESS.md
notebooks/
src/
outputs/
README.md
The two original files go in inputs/ . Do not edit them. Save the brief in Part 3 as
PROJECT_BRIEF.md in the main folder.
Part 3: The project brief(save this as PROJECT_BRIEF.md)
Copy everything in the block below into a new file.
# PROJECT BRIEF: Hybrid Quantum-Classical Term Deposit Campaign Pipeline
## Goal
A bank runs a term-deposit campaign. Build a pipeline in three linked stages.
Final answer: SIX customers to contact, and the probability that AT LEAST
THREE of them subscribe.
## Source files (in inputs/)
- PROBLEM_STATEMENT_THEATOMIQHACKTHON.pdf (official rules, 4 pages, imagebased PDF)
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 2 of 11
- bank_campaign_participant_release.xlsx with sheets:
- README: seeds and formulas
- Stage1_Train: 300 rows, columns customer_id, age, job, marital,
education, default, balance, housing, loan, contact, day, month, campaign,
pdays, previous, poutcome, y
- Stage1_Test: 100 rows, same features but NO y and NO customer_id column
(see Open Issue 1)
- Stage2_Data: 400 rows, columns customer_id, contact_cost,
expected_deposit_value, capacity_units, customer_segment
- Stage2_Constraints: the limits below
## Data facts
- Source dataset: bank.csv, 400 sampled customers. The duration field was
removed on purpose.
- Training target: 35 yes and 265 no (about 12 percent yes). Strong class
imbalance, so accuracy is NOT a useful metric.
- Seeds given by organizers: sampling seed 42, train-test split seed 42.
## Stage 1: Predict
- Train on Stage1_Train only. Predict a subscription probability (continuous,
0 to 1) for each of the 100 test customers.
- Build at least ONE quantum or hybrid probabilistic classifier (examples:
variational quantum classifier, quantum kernel classifier, quantum neural
network) AND ONE classical probabilistic baseline.
- Validate using training data only (for example stratified crossvalidation). Fit all preprocessing and any probability calibration on
training data only.
- Predict the test set once, after the pipeline is final.
- Rank test customers from highest to lowest probability. Tie-break: lower
customer_id first. The top 10 are the Stage 2 candidates.
- Judged on: ROC-AUC, PR-AUC, Precision@10, Recall@10, Brier score, log loss,
calibration.
## Stage 2: Select exactly 6 of the 10 candidates
- expected_net_value_i = predicted_probability_i * expected_deposit_value_i -
contact_cost_i
- Decision variable x_i is 0 or 1 (1 = contact).
- Maximize Z = SUM(expected_net_value_i * x_i)
- Constraints:
- SUM(x_i) = 6
- SUM(contact_cost_i * x_i) <= 25 (budget)
- SUM(capacity_units_i * x_i) <= 22 (capacity)
- Formulate as a binary optimization problem and solve with a quantum or
hybrid solver (for example QAOA or SamplingVQE).
- Explain how constraints were enforced (penalty terms, slack or auxiliary
variables). Report optimized parameters, measured bitstrings, and how they
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 3 of 11
were decoded into a selection.
- Compare with the exact classical solution (brute force over 210
combinations is enough, since C(10,6) = 210). Report the gap.
- A valid selection has exactly six unique customer IDs, all from OUR top 10,
meeting both limits. Invalid selections get no credit.
## Stage 3: Campaign outcome simulation
- Use the six selected customers and their Stage 1 probabilities.
- One qubit per customer, prepared so P(measure 1) equals that customer's
probability. Assume customers respond independently.
- Success = at least 3 qubits read 1. State the customer-to-qubit order and
handle bit-ordering correctly (Qiskit is little-endian: qubit 0 is the
rightmost bit in printed bitstrings).
- Judged against an exact calculation from our own probabilities. Also
compute a Monte Carlo estimate.
- Show ideal noiseless, finite-shot, and noisy results where possible. Report
unmitigated results alongside any mitigated ones.
## Hard rules
1. Never use the test set to select features, tune thresholds, calibrate, or
choose hyperparameters. Breaking this means a major penalty or
disqualification.
2. Fit every preprocessing step (encoders, scalers, imputers, feature
selectors) on training data only.
3. Use the same Stage 1 probabilities in Stages 2 and 3. Never hand-edit the
top 10.
4. Keep the quantum contribution clearly separate from classical reference
methods. Make no unsupported quantum-advantage claims.
5. A quantum circuit unrelated to the stated task earns no quantum credit.
6. Use customer_id exactly as given as the key in every file. One row per
customer. Probabilities between 0 and 1.
## Deliverables (single zipped folder or repo link, three result files at top
level)
- Code and notebooks that run end to end: preprocessing, models,
probabilities, Stage 2 formulation and quantum optimizer, classical
benchmark, Stage 3 circuit with exact and Monte Carlo validation.
- Stage 1 result file: all 100 test customers with customer_id, predicted
probability, rank, top-10 flag.
- Stage 2 result file: the top 10 IDs with their inputs, the six selected
IDs, budget and capacity totals, objective value, best bitstrings, classical
optimum, and the gap.
- Stage 3 result file: selected IDs, qubit order, the six probabilities,
circuit parameters, measurement counts, quantum estimate, exact result, Monte
Carlo result, and the errors.
- Short technical README: approach, validation, constraint handling, results,
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 4 of 11
error analysis, limitations, business recommendation.
- Short presentation connecting all three stages and showing what is quantum
and what is classical.
## Reproducibility requirements
- Record all seeds (data split, model, optimizer, simulator, transpiler,
Monte Carlo) and package versions.
- State the backend or simulator, its configuration, and the shot count.
- Report logical and physical qubit counts, original and transpiled circuit
depth, one-qubit and two-qubit gate counts, and SWAP count.
- README must tell judges how to rerun everything locally or in the cloud.
## Scoring (out of 100)
- 35 percent: Quantum rigor and technical depth
- 30 percent: Result accuracy and end-to-end integration (customer ID
traceability, error propagation analysis)
- 20 percent: Code quality and reproducibility
- 15 percent: Report, presentation, communication (clear quantum vs classical
split, noise and limitations)
## Open issues
1. Stage1_Test has NO customer_id column, but every result file must be keyed
by customer_id and Stage 2 needs IDs to look up cost, value, and capacity. Do
not guess the mapping. See the section on this issue in the master prompt.
Part 4: The master prompt(paste this first)
Send this once at the start of the project, with the folder already open and the two input
files already inside inputs/ .
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 5 of 11
You are my lead engineer on a hackathon project. I am a complete beginner, so
explain what you do in plain language and never skip a step.
First, read PROJECT_BRIEF.md in this folder. It contains the full rules. The
original files are in the inputs/ folder. The PDF is an image-based scan, so
use OCR or view its pages to confirm the brief matches it, and tell me if
anything differs.
Then do the following, in order:
1. Set up a Python virtual environment in this folder and install: numpy,
pandas, openpyxl, scikit-learn, matplotlib, qiskit, qiskit-aer, qiskitalgorithms, qiskit-optimization, qiskit-machine-learning. Save the exact
versions in requirements.txt.
2. Export each sheet of the Excel file to a CSV file in a new folder data/,
without changing any values.
3. Inspect the data and give me a short plain-language summary: row counts,
missing values, the class balance, and anything odd.
4. Check the customer_id problem. Stage1_Test has no customer_id column. Look
at the Excel file carefully, including hidden columns, hidden sheets, and row
order, and tell me what you find. Do NOT invent or guess an ID mapping. If
there is no safe way to link test rows to IDs, stop and write a clear note I
can send to the organizers, and wait for my answer.
5. Write a short plan for Stages 1, 2, and 3 and wait for me to say go.
Rules for this whole project:
- The test set is only for the single final prediction. Never use it for
training, tuning, feature choice, or calibration.
- Fit all preprocessing inside a pipeline on training data only.
- Use fixed random seeds (42 unless noted) and write them down.
- Keep quantum code and classical code in separate files so the split is
obvious.
- After each stage, update PROGRESS.md with what was done, what files were
produced, and what is next.
- Be honest. If a quantum model performs worse than the classical baseline,
say so. Do not claim quantum advantage.
Part 5: Stage prompts (send one at a time)
Send each prompt only after the previous stage is finished and you have read the agent's
summary.
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 6 of 11
Stage 1 prompt: predict who might subscribe
Start Stage 1. Follow PROJECT_BRIEF.md.
1. Build a preprocessing pipeline (encode categories, scale numbers) fitted
on Stage1_Train only.
2. Build a classical probabilistic baseline, such as logistic regression and
a gradient boosting model, with probability calibration done inside crossvalidation on the training data.
3. Build a quantum or hybrid classifier. Because simulators are slow, first
reduce the features to a small number (about 4 to 8) using a method fitted on
training data only, then use either a quantum kernel classifier (for example
a ZZ-type feature map with a quantum kernel and an SVM that outputs
probabilities) or a variational quantum classifier. Explain in plain language
what the circuit does.
4. Evaluate every model with stratified cross-validation on the training data
only. Report ROC-AUC, PR-AUC, Precision@10, Recall@10, Brier score, log loss,
and a calibration check. Remember the data is only about 12 percent yes, so
do not rely on accuracy.
5. Choose the final model using cross-validation results only. Explain the
choice. Then predict the 100 test customers exactly once.
6. Rank by probability (highest first), tie-break by lower customer_id, and
flag the top 10.
7. Save outputs/stage1_results.csv with customer_id, predicted_probability,
rank, top10_flag. Also save the CV metric tables and plots.
If the customer_id issue is still unresolved, stop before step 6 and remind
me.
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 7 of 11
Stage 2 prompt: pick the best six
Start Stage 2 using the top 10 from outputs/stage1_results.csv. Do not change
that list.
1. Join the 10 candidates with Stage2_Data and compute expected_net_value =
predicted_probability * expected_deposit_value - contact_cost.
2. Write the problem as a binary optimization: maximize total expected net
value with exactly 6 selected, total contact cost at most 25, total capacity
units at most 22. Show the equations and explain in plain language how each
constraint is enforced (penalty terms and slack variables). Use qiskitoptimization if helpful.
3. Solve it with a quantum or hybrid method such as QAOA or SamplingVQE on a
simulator. Record the optimizer, its settings, the optimized parameters, the
shot count, and the measured bitstrings.
4. Decode the best valid bitstring into six customer IDs. Check that the
selection has exactly six unique IDs, all from the top 10, and meets both
limits.
5. Solve the same problem exactly by brute force over all 210 combinations
and compare: objective value, whether the quantum answer matched, and the
gap.
6. Save outputs/stage2_results.csv or .json with all the fields listed in
PROJECT_BRIEF.md.
If the quantum solver often returns invalid selections, show me how often,
and try stronger penalties or more layers before giving up. Report honestly
what happened.
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 8 of 11
Stage 3 prompt: simulate the campaign
Start Stage 3 using the six selected customers and their Stage 1
probabilities.
1. Build a 6-qubit circuit with one qubit per customer. Prepare each qubit
with a rotation so the chance of measuring 1 equals that customer's
probability. Write the customer-to-qubit order down and handle Qiskit bit
ordering correctly (qubit 0 is the rightmost character in printed
bitstrings).
2. Count a shot as a success when at least 3 qubits read 1.
3. Compute the exact probability of at least 3 successes from the six
probabilities (independent customers). Also run a classical Monte Carlo
estimate.
4. Run the circuit on an ideal simulator with many shots, and report the
estimate and error versus the exact value. Repeat with a few different shot
counts and seeds to show how the error shrinks.
5. Add a noisy simulation using a simple noise model and report the result.
If you apply error mitigation, show unmitigated and mitigated results side by
side.
6. Report logical and physical qubit counts, original and transpiled depth,
one-qubit and two-qubit gate counts, and SWAP count.
7. Write an error propagation analysis: how much would the final probability
change if the Stage 1 probabilities were off by a small amount?
8. Save outputs/stage3_results.csv or .json with all the fields listed in
PROJECT_BRIEF.md.
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 9 of 11
Packaging prompt: finish the submission
Package the project for submission.
1. Make sure all code runs from a clean start. Run it end to end once and fix
anything that breaks.
2. Confirm the three result files are at the top level of the final folder:
stage1_results, stage2_results, stage3_results.
3. Write README.md with: approach, how to run everything, validation method,
how the constraints were handled, results for all three stages, error
analysis, limitations, the quantum vs classical split, and a short business
recommendation for the bank.
4. Record all seeds, package versions, backend or simulator details, and shot
counts.
5. Check every rule in PROJECT_BRIEF.md one by one and give me a yes or no
for each, with proof.
6. Create a clean zipped folder ready to submit.
7. Draft a 6 to 8 slide presentation outline that tells one story across the
three stages and clearly marks what is quantum and what is classical. Keep
the wording simple enough for me to present.
Part 6: Final checklist before you submit
The test set was never used for training, tuning, feature choice, or calibration.
Stage 1 file has 100 rows, with customer_id, probability between 0 and 1, rank, and top10 flag.
Stage 2 selection has exactly six unique IDs, all from your own top 10, cost at most 25,
capacity at most 22.
Stage 2 quantum result is compared with the brute-force optimum, and the gap is
reported.
Stage 3 uses the same Stage 1 probabilities and states the qubit order.
Stage 3 shows exact, Monte Carlo, ideal quantum, and noisy results, with errors.
The README says plainly what is quantum and what is classical, and makes no
unsupported quantum-advantage claims.
Seeds, package versions, simulator details, shot counts, qubit counts, circuit depth,
gate counts, and SWAP count are all recorded.
The code runs from a clean start using the README instructions.
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 10 of 11
Part 7: Questions to send the organizers
Send these through the channel the organizers announced. Do this early, before building
Stage 1 outputs.
1. Stage1_Test has no customer_id column. How should the 100 test rows be matched to
customer_ids? Is the row order tied to the IDs that are missing from Stage1_Train?
2. Is any specific quantum SDK or simulator required, or is Qiskit with a local simulator
acceptable?
3. Is there a limit on the number of qubits or circuit depth for the Stage 1 quantum
model?
Part 8: Common problems and quick fixes
Problem What to do
The agent wants to use the test data to pick
settings
Say no. Remind it of hard rule 1 in
PROJECT_BRIEF.md.
Installing a package fails Paste the full error to the agent and ask it to
fix the environment.
The quantum model is very slow Ask the agent to reduce the number of
features and qubits, and use a smaller
training subset for the circuit search while
keeping the rules intact.
The quantum model scores worse than the
classical one
That is fine and common on small data.
Report it honestly and explain why. Honest
analysis scores better than hidden problems.
The agent says something you do not
understand
Ask it to explain again as if you were new to
the topic.
You start a new chat and the agent forgets Tell it to read PROJECT_BRIEF.md and
PROGRESS.md first, then continue.
Atomiq Hackathon: Claude Handoff Pack and Antigravity Guide
Page 11 of 11