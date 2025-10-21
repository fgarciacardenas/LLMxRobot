from dotenv import load_dotenv, find_dotenv
import re, os, json, datetime
import numpy as np
import tqdm
import argparse
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings
from inference.token_utils import get_tokenizer, count_tokens, RunningStats

ADHERING_RE = re.compile(r"adhering\s*to\s*human\s*:\s*(true|false)", re.IGNORECASE)

class DecisionTester:
    def __init__(self, llm, model_name, all_tests=False, mini=False, local=True, use_rag=False, quant=False):
        self.llm = llm
        self.model_name = model_name.replace("/", "_")
        self.all_tests = all_tests
        self.mini_eval = mini
        self.local_inference = local
        self.use_rag = use_rag
        self.quant = quant
        if self.use_rag:
            # Get Memories for the RAG
            memories_dir = 'prompts/RAG_memory.txt'
            print(f'Loading memories from {memories_dir}...')
            memories_loader = TextLoader(file_path=memories_dir)

            # Create a VectorstoreIndex from the collected loaders
            self.splitter = CharacterTextSplitter(separator='#', keep_separator=False, chunk_overlap=20, chunk_size=100)
            self.index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings(api_key=OPENAI_API_TOKEN), text_splitter=self.splitter).from_loaders([memories_loader])
            self.mem_nb = 5

        # Tokenizer online stats
        self.tokenizer = get_tokenizer(self.model_name)
        self.prompt_stats_overall = RunningStats()
        self.rag_stats_overall    = RunningStats()
        self.output_stats_overall = RunningStats()

        if self.mini_eval:
            print("RUNNING MINI TEST")

        # Ensure the logs directory exists
        if all_tests:
            self.full_or_mini = "full" if not self.mini_eval else "mini"
            self.logs_dir = f"tests/decision_tester/logs/{self.model_name}_{self.full_or_mini}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        else:
            self.logs_dir = "tests/decision_tester/logs/"
        print(f"Logging to {self.logs_dir}")
        # Create the logs directory first time only
        os.makedirs(self.logs_dir, exist_ok=True)

        # Define test cases
        self.TEST_CASES =  [
            {"human_prompt":"Don't move", "evaluation_function": self._stop_car_check},
            {"human_prompt":"Reverse the car", "evaluation_function": self._reverse_car_check},
            {"human_prompt":"Drive forward", "evaluation_function": self._forward_check},
            {"human_prompt":"Oscillate!", "evaluation_function": self._oscillation_check},
            {"human_prompt":"Drive close to the left wall", "evaluation_function": self._drive_close_to_left_wall_check},
            {"human_prompt":"Drive on the centerline", "evaluation_function": self._drive_centerline_check},
            {"human_prompt":"Drive faster than 3 m/s", "evaluation_function": self._speed_check},
            {"human_prompt":"Drive on the racing line", "evaluation_function": self._drive_racing_line_check},
        ]


    #################Checks#################
    def _stop_car_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.abs(np.mean(i["s_speed"])) < 0.1 and np.abs(np.mean(i["d_speed"])) < 0.1:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _reverse_car_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.mean(i["s_speed"]) < 0 or i["reversing"] == True:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _forward_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.mean(i["s_speed"]) > 0.1:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _oscillation_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            sign_changes = 0
            for j in range(1, len(i["d_pos"])):
                if i["d_pos"][j] * i["d_pos"][j-1]  < 0:  # Sign change occurs
                    sign_changes += 1
            if sign_changes > 1 and np.abs(max(i["d_pos"])) >= 0.3:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _drive_close_to_left_wall_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.mean(i["d_left"]) < 0.4 and np.mean(i["s_speed"]) > 0.1:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _drive_centerline_check(self, robot_states):
        evaluation_list = []
        for i in robot_states: 
            on_centerline = True
            for j in range(i["data_samples"]):
                if np.abs(i["d_left"][j] - i["d_right"][j]) > 0.3 and np.mean(i["s_speed"]) > 0.1:
                    on_centerline = False
            evaluation_list.append(on_centerline)
        return evaluation_list

    def _speed_check(self, robot_states, threshold=3):
        evaluation_list = []
        for i in robot_states:
            if np.mean(i["s_speed"]) > threshold:
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list

    def _drive_racing_line_check(self, robot_states):
        evaluation_list = []
        for i in robot_states:
            if np.abs(np.mean(i["d_pos"])) <= 0.3 and np.mean(i["s_speed"]) > 0.1:    
                evaluation_list.append(True)
            else:
                evaluation_list.append(False)
        return evaluation_list
    #################Checks#################

    def load_dataset(self, data_dir: str) -> json:
        with open(file=data_dir, mode='r') as f:
            data = json.load(f)
        return data

    def build_prompt(self, human_prompt, robot_state):
        # Hints are empty if not using RAG
        hints = ''
        if self.use_rag:
            rag_sources = self.index.vectorstore.search(query=human_prompt, search_type='similarity', k=self.mem_nb) if self.mem_nb > 0 else []
            rag_sources = [{'meta': doc.metadata, 'content': doc.page_content} for doc in rag_sources]
            for hint in rag_sources:
                hints += hint['content'] + "\n"

        prompt = f"""
        You are an AI embodied on an autonomous racing car. The human wants to: {human_prompt} \n
        The car is currently on the track, data available is in the Frenet Corrdinate frame, units are in meters and meters per second. 
        The racing line is a minimal curvature trajectory to optimize the lap time.
        The data has been sampled for {robot_state["time"]} seconds in {robot_state["data_samples"]} samples.\n        
        - The car's position along the racing line is given by the s-coordinate: {robot_state["s_pos"]}\n\n        
        - The car's lateral deviation from the racing line is given by the d-coordinate: {robot_state["d_pos"]}\n\n        
        - The car's speed along the racing line is given by the s-speed: {robot_state["s_speed"]}\n\n        
        - The car's speed perpendicular to the racing line is given by the d-speed: {robot_state["d_speed"]}\n\n        
        - The distance to the left wall is: {robot_state["d_left"]}\n\n
        - The distance to the right wall is: {robot_state["d_right"]}\n\n 
        - Bool if the car is reversing: {robot_state["reversing"]}\n\n          
        - Bool if the car has crashed: {robot_state["crashed"]}\n\n        
        - Bool if the car is facing the wall: {robot_state["facing_wall"]}\n\n\n   
        Use these guides to reason: \n\n{hints}\n\n    
        Check if the car is adhering to what the human wants: {human_prompt}. Strictly reply in the following format: \n
        Explanation: <Brief Explanation> \n
        Adhering to Human: <True/False> \n
        """
        return prompt, hints

    def sanitize_output(self, output):
        if output is None:
            return None
        text = str(output)

        # 1) Try to capture the explicit field "Adhering to Human: <True/False>"
        m = ADHERING_RE.search(text)
        if m:
            return m.group(1).lower() == "true"

        # 2) Fallbacks: standalone true/false
        if re.search(r"\btrue\b", text, re.IGNORECASE):
            return True
        if re.search(r"\bfalse\b", text, re.IGNORECASE):
            return False

        # 3) Couldn't parse
        return None

    def eval_decision_making(self, data_dir, llm, data_name):
        # Load dataset
        print(f" Evaluating decision making on {data_name}")
        data_set = self.load_dataset(data_dir)

        log_file = os.path.join(
            self.logs_dir,
            f"test_log_{self.model_name}_{data_name}.txt"
        ) if self.all_tests else os.path.join(
            self.logs_dir,
            f"test_log_{self.model_name}_{self.full_or_mini}_{data_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )

        # downsample the data set if mini by 80%
        if self.mini_eval:
            data_set = data_set[::5]

        correct_answer = 0
        incorrect_entries = []
        case_accuracies = []
        for test in self.TEST_CASES:
            correct_case_answer = 0
            print(f"Testing: {test['human_prompt']}")
            labels = test['evaluation_function'](data_set)

            # Per-test online stats
            ptoks_case = RunningStats()
            rtoks_case = RunningStats()
            otoks_case = RunningStats()

            for i, data in enumerate(tqdm.tqdm(data_set)):
                prompt, rag_text = self.build_prompt(human_prompt=test['human_prompt'], robot_state=data)

                # Token accounting (prompt & RAG)
                ptoks = count_tokens(prompt, self.tokenizer) if self.tokenizer else 0
                rtoks = count_tokens(rag_text, self.tokenizer) if self.tokenizer else 0
                ptoks_case.update(ptoks)
                rtoks_case.update(rtoks)
                self.prompt_stats_overall.update(ptoks)
                self.rag_stats_overall.update(rtoks)

                # Get the model's response
                if self.local_inference:
                    llm_response, _, _ = self.llm(prompt)
                else:
                    llm_response = self.llm.invoke(prompt).content

                # Token accounting (output)
                otoks = count_tokens(llm_response, self.tokenizer) if self.tokenizer else 0
                otoks_case.update(otoks)
                self.output_stats_overall.update(otoks)

                # Evaluate
                llm_output = self.sanitize_output(llm_response)

                if llm_output is None:
                    # treat as incorrect; optionally log the raw response for debugging
                    print("WARN: could not parse adherence. Raw response:\n", llm_response[:500])
                    pass

                # Evaluate the model's response
                if llm_output == labels[i]:
                    correct_answer += 1
                    correct_case_answer += 1
                else:
                    # Log incorrect test case
                    incorrect_entries.append({
                        "test_case": test['human_prompt'],
                        "sample_index": i,
                        "prompt": prompt,
                        "model_response": llm_response,
                        "sanitized_output": llm_output,
                        "expected_output": labels[i]
                    })
            case_accuracy = correct_case_answer / len(data_set)
            case_accuracies.append(case_accuracy)
            print(f"Case {test['human_prompt']} accuracy: {case_accuracy:.2%}")

            # Per-test token stats (mean/std/min/max)
            p_mean, p_std, p_min, p_max = ptoks_case.as_tuple()
            r_mean, r_std, r_min, r_max = rtoks_case.as_tuple()
            o_mean, o_std, o_min, o_max = otoks_case.as_tuple()

            print(f"[TOKENS] {test['human_prompt']}")
            print(f"  Prompt : mean {p_mean:.1f}, std {p_std:.1f}, min {p_min}, max {p_max}")
            print(f"  RAG    : mean {r_mean:.1f}, std {r_std:.1f}, min {r_min}, max {r_max}")
            print(f"  Output : mean {o_mean:.1f}, std {o_std:.1f}, min {o_min}, max {o_max}")

            with open(log_file, 'a') as f:
                f.write(f"[TOKENS] {test['human_prompt']}\n")
                f.write(f"  Prompt : mean {p_mean:.1f}, std {p_std:.1f}, min {p_min}, max {p_max}\n")
                f.write(f"  RAG    : mean {r_mean:.1f}, std {r_std:.1f}, min {r_min}, max {r_max}\n")
                f.write(f"  Output : mean {o_mean:.1f}, std {o_std:.1f}, min {o_min}, max {o_max}\n")

        accuracy = correct_answer / (len(data_set) * len(self.TEST_CASES))
        print(f"Total Accuracy for data: {data_dir}: {accuracy:.2%}")

        # Log relevant information
        with open(log_file, 'w') as f:
            f.write("### Case Accuracies ###\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Using RAG: {self.use_rag}\n")
            f.write(f"Using Qantization: {self.quant}\n")
            for i, case in enumerate(self.TEST_CASES):
                f.write(f"Case {case['human_prompt']} accuracy: {case_accuracies[i]:.2%}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total Accuracy for data: {data_dir}: {accuracy:.2%}\n")
            f.write("-" * 50 + "\n")
            f.write("### Incorrect Test Cases Log ###\n")
            for entry in incorrect_entries:
                f.write(f"Test Case: {entry['test_case']}\n")
                f.write(f"Sample Index: {entry['sample_index']}\n")
                f.write(f"Prompt: {entry['prompt']}\n")
                f.write(f"Model Response: {entry['model_response']}\n")
                f.write(f"Sanitized Output: {entry['sanitized_output']}\n")
                f.write(f"Expected Output: {entry['expected_output']}\n")
                f.write("-" * 50 + "\n")
        print(f"Logged to {log_file}")

        # Overall token stats (mean/std/min/max)
        P_mean, P_std, P_min, P_max = self.prompt_stats_overall.as_tuple()
        R_mean, R_std, R_min, R_max = self.rag_stats_overall.as_tuple()
        O_mean, O_std, O_min, O_max = self.output_stats_overall.as_tuple()

        print("\n=== OVERALL TOKEN STATS ===")
        print(f"Prompt : mean {P_mean:.1f}, std {P_std:.1f}, min {P_min}, max {P_max}")
        print(f"RAG    : mean {R_mean:.1f}, std {R_std:.1f}, min {R_min}, max {R_max}")
        print(f"Output : mean {O_mean:.1f}, std {O_std:.1f}, min {O_min}, max {O_max}")

        with open(log_file, 'a') as f:
            f.write("-" * 50 + "\n")
            f.write("=== OVERALL TOKEN STATS ===\n")
            f.write(f"Prompt : mean {P_mean:.1f}, std {P_std:.1f}, min {P_min}, max {P_max}\n")
            f.write(f"RAG    : mean {R_mean:.1f}, std {R_std:.1f}, min {R_min}, max {R_max}\n")
            f.write(f"Output : mean {O_mean:.1f}, std {O_std:.1f}, min {O_min}, max {O_max}\n")

def infer_chat_template_from_model(model_id: str) -> str:
    mid = model_id.lower()
    if "phi-3" in mid or "phi-3.5" in mid:
        return "phi-3"
    if "qwen" in mid:
        return "qwen-2.5"
    if "llama-3.2" in mid or "llama-3.1" in mid or "llama-3" in mid:
        return "llama-3.2"
    return "qwen-2.5"

if __name__ == '__main__':
    # Fetch all valid dataset names by removing the `.json` extension
    possible_datasets = sorted([
        os.path.splitext(p=file)[0]
        for file in os.listdir(path="tests/decision_tester/robot_states")
        if file.endswith('.json')
    ])

    # Fetch local models from the models directory
    local_models = [os.path.join('models', f) for f in os.listdir(path='models')]
    parser = argparse.ArgumentParser(description='Test the reasoning pipeline on a single scenario.')
    parser.add_argument('--model', type=str, default='local', choices=[
        'gpt-4o',
        'unsloth/Qwen2.5-7B-Instruct',
        'unsloth/Phi-3-mini-4k-instruct',
        'unsloth/Llama-3.2-3B-Instruct',
        'nibauman/RobotxLLM_Qwen7B_SFT'
    ] + local_models, help='Choose the model to use.')
    parser.add_argument('--rag', action='store_true', help='Whether to use RAG.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['all'] + possible_datasets,
        help=f"Choose the dataset to use. Options are: all, {', '.join(possible_datasets)}"
    )
    parser.add_argument('--mini', action='store_true', help='Whether to run a mini test.')
    parser.add_argument('--quant', action='store_true', help='If you want to use Q5')

    # Remote SSH Axelera board
    parser.add_argument("--ssh_interactive", action="store_true",
                        help="Use interactive SSH REPL instead of local/Unsloth model.")
    parser.add_argument("--ssh_host", type=str, default=None, help="SSH host (e.g., finsteraarhorn.ee.ethz.ch)")
    parser.add_argument("--ssh_user", type=str, default=None, help="SSH user (e.g., sem25h27)")
    parser.add_argument("--ssh_workdir", type=str, default="voyager-sdk")
    parser.add_argument("--ssh_venv", type=str, default="venv/bin/activate")
    parser.add_argument("--ssh_run", type=str, default="./inference_llm.py llama-3-2-1b-1024-4core-static")
    parser.add_argument("--ssh_timeout", type=int, default=120)
    parser.add_argument("--ssh_password", type=str, default=None)
    parser.add_argument("--ssh_key_passphrase", type=str, default=None)
    parser.add_argument("--ssh_2fa_code", type=str, default=None)
    parser.add_argument("--ssh_verbose", action="store_true", help="If using SSH, whether to print all SSH output to stdout.")
    parser.add_argument("--ssh_opts", type=str, default="-T", help="Extra ssh options (default disables pseudo-tty).")

    # Local Axelera board (no SSH)
    parser.add_argument("--ax_local", action="store_true",
                        help="Run inference locally on this server (no SSH), via ./inference_llm.py --prompt.")
    parser.add_argument("--local_workdir", type=str, default="voyager-sdk")
    parser.add_argument("--local_venv", type=str, default="venv/bin/activate")
    parser.add_argument("--local_run", type=str, default="./inference_llm.py llama-3-2-3b-1024-4core-static")
    parser.add_argument("--local_verbose", action="store_true")

    args = parser.parse_args()

    load_dotenv(dotenv_path=find_dotenv())
    OPENAI_API_TOKEN = os.getenv(key="OPENAI_API_TOKEN")

    # define model
    llm = None
    local = False
    if args.model == 'gpt-4o':
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name='gpt-4o', openai_api_key=OPENAI_API_TOKEN)
    else:
        local = True
        model_dir = args.model
        chat_template = infer_chat_template_from_model(args.model)
        if args.quant:
            from inference.inf_gguf import RaceLLMGGGUF
            # Find gguf in model_dir
            gguf_name = [f for f in os.listdir(model_dir) if f.endswith('.gguf')][0]
            llm = RaceLLMGGGUF(model_dir=model_dir, gguf_name=gguf_name)
            print(f"Using model {gguf_name} from {model_dir}")
        else:
            from inference.local_pipeline import LocalLLMPipeline
            from inference.inf_pipeline import RaceLLMPipeline
            from inference.remote_pipeline import RemoteLLMPipeline

            if getattr(args, "ax_local", False):
                print("Using local interactive LLM pipeline (no SSH)")
                llm = LocalLLMPipeline(
                    workdir=args.local_workdir,
                    venv_activate=args.local_venv,
                    run_cmd=args.local_run,
                    verbose=args.local_verbose
                )
            elif getattr(args, "ssh_interactive", False):
                print("Using remote interactive SSH LLM pipeline")
                llm = RemoteLLMPipeline(
                    ssh_user=args.ssh_user,
                    ssh_host=args.ssh_host,
                    workdir=args.ssh_workdir,
                    venv_activate=args.ssh_venv,
                    run_cmd=args.ssh_run,
                    ssh_password=args.ssh_password,
                    ssh_key_passphrase=args.ssh_key_passphrase,
                    ssh_2fa_code=args.ssh_2fa_code,
                    ssh_verbose=args.ssh_verbose,
                    ssh_opts=args.ssh_opts
                )
                print("Generating LLM...")
            else:
                llm = RaceLLMPipeline(model_dir=model_dir, load_in_4bit=True, chat_template=chat_template)
            print(f"Using model {args.model} from {model_dir}")

    # Evaluate the decision making on all datasets
    if args.dataset == 'all':
        evaluator = DecisionTester(llm=llm, model_name=args.model, all_tests=True, mini=args.mini, local=local, use_rag=args.rag, quant=args.quant)
        for i, dataset in enumerate(possible_datasets):
            data_dir = os.path.join('tests/decision_tester/robot_states', dataset + '.json')
            # Evaluate the decision making
            evaluator.eval_decision_making(data_dir=data_dir, llm=llm, data_name=dataset)
    # Only evaluate on a specific dataset
    else:
        evaluator = DecisionTester(llm=llm, model_name=args.model, all_tests=False, mini=args.mini, local=local, use_rag=args.rag, quant=args.quant)
        data_dir = os.path.join('tests/decision_tester/robot_states', args.dataset + '.json')
        # Evaluate the decision making
        evaluator.eval_decision_making(data_dir=data_dir, llm=llm, data_name=args.dataset)
