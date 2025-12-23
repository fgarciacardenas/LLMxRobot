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
from inference.rag_offline import OfflineRetriever, LocalEmbeddings

ADHERING_RE = re.compile(r"^\s*adhering\s*to\s*human\s*:\s*(true|false)", re.IGNORECASE)
ADHERING_ANY_RE = re.compile(r"adhering\s*to\s*human\s*:\s*(true|false)", re.IGNORECASE)

class DecisionTester:
    def __init__(
        self,
        llm=None,
        model_name=None,
        tokenizer=None,
        all_tests=False,
        mini=False,
        local=True,
        use_rag=False,
        quant=False,
        rag_offline=False,
        rag_index="",
        rag_corpus="prompts",
        rag_max_hits=5,
        rag_score_threshold=0.0,
        rag_fetch_k=5,
        binary_output=False,
        openai_api_token=None,
        **kwargs,
    ):
        # Backwards-compatible aliasing for older call sites.
        if openai_api_token is None:
            openai_api_token = kwargs.pop("OPENAI_API_TOKEN", None)
        else:
            kwargs.pop("OPENAI_API_TOKEN", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"DecisionTester.__init__() got unexpected keyword argument(s): {unexpected}")

        self.openai_api_token = (
            openai_api_token
            or os.getenv("OPENAI_API_TOKEN")
            or os.getenv("OPENAI_API_KEY")
        )

        self.llm = llm
        if model_name is None:
            raise TypeError("DecisionTester.__init__() missing required argument: 'model_name'")
        self.model_name = model_name.replace("/", "_")
        self.all_tests = all_tests
        self.mini_eval = mini
        # Used in log naming for non-all_tests runs as well.
        self.full_or_mini = "mini" if self.mini_eval else "full"
        self.local_inference = local
        self.use_rag = use_rag
        self.quant = quant
        self.rag_offline = rag_offline
        self.rag_index = rag_index
        self.rag_corpus = rag_corpus
        self.rag_max_hits = rag_max_hits
        self.rag_score_threshold = rag_score_threshold
        self.rag_fetch_k = rag_fetch_k if rag_fetch_k is not None else 5
        self.binary_output = binary_output

        if self.use_rag:
            self.mem_nb = self.rag_fetch_k if self.rag_fetch_k is not None else 5
            
            if self.rag_offline:
                self._init_rag_offline()
            else:
                if not self.openai_api_token:
                    raise ValueError(
                        "RAG (online) requires an OpenAI token; set `OPENAI_API_TOKEN`/`OPENAI_API_KEY` "
                        "or pass `openai_api_token=` to DecisionTester()."
                    )
                # Get Memories for the RAG
                memories_dir = 'prompts/RAG_memory.txt'
                print(f'Loading memories from {memories_dir}...')
                memories_loader = TextLoader(file_path=memories_dir)

                # Create a VectorstoreIndex from the collected loaders
                self.splitter = CharacterTextSplitter(
                    separator='#', keep_separator=False, chunk_overlap=20, chunk_size=100
                )
                self.index = VectorstoreIndexCreator(
                    embedding=OpenAIEmbeddings(api_key=self.openai_api_token),
                    text_splitter=self.splitter
                ).from_loaders([memories_loader])

        # Tokenizer online stats
        self.tokenizer = tokenizer or get_tokenizer(self.model_name)
        self.prompt_stats_overall = RunningStats()
        self.rag_stats_overall    = RunningStats()
        self.output_stats_overall = RunningStats()

        if self.mini_eval:
            print("RUNNING MINI TEST")

        # Ensure the logs directory exists
        if all_tests:
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

    def _init_rag_offline(self):
        """
        Initializes an OfflineRetriever.
        - If a prebuilt index prefix is provided via --rag_index and exists, load it.
        - Else, build from the same source you already use: prompts/RAG_memory.txt (split by '#')
        """
        print("[RAG] Using OFFLINE embeddings")
        index_prefix = self.rag_index
        if index_prefix and os.path.exists(index_prefix + ".meta.pkl"):
            print(f"[RAG] Loading offline index from {index_prefix}.*")
            self.offline_retriever = OfflineRetriever(index_path=index_prefix, embeddings=LocalEmbeddings())
            self.offline_retriever.load()
            return

        # Build on the fly from the same memory file you already use
        memories_path = os.path.join(self.rag_corpus, "RAG_memory.txt")
        if not os.path.exists(memories_path):
            raise FileNotFoundError(
                f"[RAG] Could not find {memories_path}. "
                "Provide --rag_index pointing to a prebuilt index or ensure prompts/RAG_memory.txt exists."
            )
        print(f"[RAG] Building offline index from {memories_path}")
        loader = TextLoader(file_path=memories_path)
        docs = loader.load()
        splitter = CharacterTextSplitter(separator='#', keep_separator=False, chunk_overlap=20, chunk_size=100)
        chunks = splitter.split_documents(docs)
        texts = [c.page_content for c in chunks if c and c.page_content]
        ids = [f"mem_{i}" for i in range(len(texts))]

        # Store in a session-scoped index path
        os.makedirs("data/rag_index", exist_ok=True)
        idx_prefix = os.path.join("data", "rag_index", "session_offline")
        self.offline_retriever = OfflineRetriever(index_path=idx_prefix, embeddings=LocalEmbeddings())
        self.offline_retriever.build(texts, ids)

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

    def _gather_rag_candidates(self, human_prompt):
        """
        Retrieve raw RAG candidates (with scores when available) without filtering.
        """
        if not self.use_rag:
            return []

        fetch_k = self.rag_fetch_k if self.rag_fetch_k is not None else 0
        if fetch_k <= 0:
            return []

        candidates = []
        if self.rag_offline:
            hits = self.offline_retriever.retrieve(human_prompt, k=fetch_k)
            for (_id, txt, score) in hits:
                candidates.append({
                    "id": _id,
                    "source": "offline",
                    "text": txt or "",
                    "score": float(score) if score is not None else None,
                })
        else:
            vectorstore = getattr(self.index, "vectorstore", None)
            docs_with_scores = []
            if vectorstore is not None and hasattr(vectorstore, "similarity_search_with_score"):
                docs_with_scores = vectorstore.similarity_search_with_score(human_prompt, k=fetch_k)
            else:
                docs = []
                if vectorstore is not None:
                    if hasattr(vectorstore, "search"):
                        docs = vectorstore.search(query=human_prompt, search_type='similarity', k=fetch_k)
                    elif hasattr(vectorstore, "similarity_search"):
                        docs = vectorstore.similarity_search(human_prompt, k=fetch_k)
                docs_with_scores = [(doc, None) for doc in (docs or [])]

            for doc_score in docs_with_scores:
                if isinstance(doc_score, tuple) and len(doc_score) == 2:
                    doc, score = doc_score
                else:
                    doc, score = doc_score, None
                meta = getattr(doc, "metadata", {}) or {}
                content = getattr(doc, "page_content", "") or ""
                candidates.append({
                    "metadata": meta,
                    "source": meta.get("source"),
                    "text": content,
                    "score": float(score) if score is not None else None,
                })
        return candidates

    def _filter_rag_candidates(self, candidates):
        """
        Apply similarity thresholding and max hint limits to raw candidates.
        """
        if not candidates:
            return []

        filtered = []
        threshold = self.rag_score_threshold if self.rag_score_threshold is not None else None
        for hit in candidates:
            score = hit.get("score")
            if threshold is not None and score is not None and score < threshold:
                continue
            filtered.append(hit)

        if self.rag_max_hits and self.rag_max_hits > 0:
            filtered = filtered[:self.rag_max_hits]

        return filtered

    def build_prompt(self, human_prompt, robot_state):
        # Hints are empty if not using RAG
        hints = ''
        rag_details = []
        rag_candidates = []
        if self.use_rag:
            rag_candidates = self._gather_rag_candidates(human_prompt)
            rag_details = self._filter_rag_candidates(rag_candidates)
            hint_texts = [hit.get("text") for hit in rag_details if hit.get("text")]
            if hint_texts:
                hints = "\n".join(hint_texts) + "\n"

        if self.binary_output:
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
        Adhering to Human: <True/False> \n
        """
        else:
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
#         hints_block = f"HINTS\n{hints}" if hints else ""

#         prompt = f"""Task: Decide if the car follows the human command.

# Command: {human_prompt}
# Frame: Frenet (m, m/s). Duration: {robot_state['time']} s. Samples: {robot_state['data_samples']}

# DATA
# s_pos={robot_state['s_pos']}
# d_pos={robot_state['d_pos']}
# s_speed={robot_state['s_speed']}
# d_speed={robot_state['d_speed']}
# d_left={robot_state['d_left']}
# d_right={robot_state['d_right']}
# reversing={robot_state['reversing']}
# crashed={robot_state['crashed']}
# facing_wall={robot_state['facing_wall']}

# {hints_block}

# Return exactly:
# Explanation: <brief>
# Adhering to Human: <True/False>"""

        return prompt, hints, rag_details, rag_candidates

    def sanitize_output(self, output):
        if output is None:
            return None
        text = str(output)

        # 1) Try to capture the explicit field "Adhering to Human: <True/False>"
        if self.binary_output:
            m = ADHERING_RE.search(text)
            if m:
                return m.group(1).lower() == "true"
            return None
        else:
            m = ADHERING_ANY_RE.search(text)
            if m:
                return m.group(1).lower() == "true"

        # Fallbacks for non-binary mode: standalone true/false anywhere
        if not self.binary_output:
            if re.search(r"\btrue\b", text, re.IGNORECASE):
                return True
            if re.search(r"\bfalse\b", text, re.IGNORECASE):
                return False

        # Couldn't parse in the expected format
        return None

    def eval_decision_making(self, data_dir, llm, data_name):
        effective_llm = llm or self.llm
        if effective_llm is None:
            raise ValueError("No LLM provided: pass `llm=` to eval_decision_making() or `llm=` to DecisionTester().")
        self.llm = effective_llm

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
        sample_debug_entries = []
        for test in self.TEST_CASES:
            correct_case_answer = 0
            print(f"Testing: {test['human_prompt']}")
            labels = test['evaluation_function'](data_set)

            # Per-test online stats
            ptoks_case = RunningStats()
            rtoks_case = RunningStats()
            otoks_case = RunningStats()

            for i, data in enumerate(tqdm.tqdm(data_set)):
                (
                    prompt,
                    rag_text,
                    rag_details,
                    rag_candidates,
                ) = self.build_prompt(human_prompt=test['human_prompt'], robot_state=data)

                # Token accounting (prompt & RAG)
                ptoks = count_tokens(prompt, self.tokenizer) if self.tokenizer else 0
                rtoks = count_tokens(rag_text, self.tokenizer) if self.tokenizer else 0
                ptoks_case.update(ptoks)
                rtoks_case.update(rtoks)
                self.prompt_stats_overall.update(ptoks)
                self.rag_stats_overall.update(rtoks)

                # Get the model's response
                if self.local_inference:
                    result = self.llm(prompt)
                    llm_response = result[0] if isinstance(result, (tuple, list)) else result
                else:
                    # prompt = " ".join(prompt.split())
                    llm_response = self.llm.invoke(prompt).content

                # Token accounting (output)
                otoks = count_tokens(llm_response, self.tokenizer) if self.tokenizer else 0
                otoks_case.update(otoks)
                self.output_stats_overall.update(otoks)

                # Evaluate
                llm_output = self.sanitize_output(llm_response)
                structure_followed = llm_output is not None

                rag_mode = "offline" if (self.use_rag and self.rag_offline) else ("online" if self.use_rag else "disabled")
                sample_debug_entries.append({
                    "test_case": test['human_prompt'],
                    "sample_index": i,
                    "rag_mode": rag_mode,
                    "rag_k": self.mem_nb if self.use_rag else 0,
                    "rag_threshold": self.rag_score_threshold if self.use_rag else None,
                    "rag_max_hits": self.rag_max_hits if self.use_rag else None,
                    "rag_candidate_count": len(rag_candidates),
                    "rag_used_count": len(rag_details),
                    "rag_details": rag_details,
                    "rag_candidates": rag_candidates,
                    "rag_text": rag_text,
                    "prompt": prompt,
                    "model_response_raw": llm_response,
                    "structure_followed": structure_followed,
                    "sanitized_output": llm_output,
                    "expected_output": labels[i],
                    "prompt_tokens": ptoks,
                    "rag_tokens": rtoks,
                    "output_tokens": otoks,
                    "model_name": self.model_name,
                })

                if llm_output is None:
                    # treat as incorrect; log structure issues
                    print("WARN: Structure not followed. Raw response:\n", llm_response[:500])
                    incorrect_entries.append({
                        "test_case": test['human_prompt'],
                        "sample_index": i,
                        "prompt": prompt,
                        "model_response": llm_response,
                        "sanitized_output": None,
                        "expected_output": labels[i],
                        "note": "Structure not followed"
                    })
                    continue

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

        debug_log_file = (
            log_file.replace(".txt", "_samples.json")
            if log_file.endswith(".txt")
            else f"{log_file}_samples.json"
        )
        with open(debug_log_file, 'w') as f:
            for entry in sample_debug_entries:
                f.write(json.dumps(entry))
                f.write("\n")
        print(f"Sample-level debug logged to {debug_log_file}")

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
    
    # Model parameters
    parser.add_argument('--model', type=str, default='local', choices=[
        'gpt-4o',
        'unsloth/Qwen2.5-7B-Instruct',
        'microsoft/Phi-3-mini-4k-instruct',
        'unsloth/Phi-3-mini-4k-instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'unsloth/Llama-3.2-3B-Instruct',
        'nibauman/RobotxLLM_Qwen7B_SFT'
    ] + local_models, help='Choose the model to use.')
    parser.add_argument('--quant', action='store_true', help='If you want to use Q5')
    
    # RAG arguments
    parser.add_argument('--rag', action='store_true', help='Whether to use RAG.')
    parser.add_argument("--rag_offline", action="store_true",
                        help="Use offline embeddings instead of OpenAI for RAG")
    parser.add_argument("--rag_index", type=str, default="",
                        help="Path prefix of offline index (e.g., data/rag_index/offline)")
    parser.add_argument("--rag_corpus", type=str, default="prompts",
                        help="Directory with .txt/.md/.json to build an index if none is provided")
    parser.add_argument("--rag_max_hits", type=int, default=5,
                        help="Maximum number of hints to include after filtering (<=0 keeps all).")
    parser.add_argument("--rag_threshold", type=float, default=0.0,
                        help="Minimum similarity score required to include a hint.")
    parser.add_argument("--rag_fetch_k", type=int, default=5,
                        help="Number of candidates to retrieve before filtering.")
    parser.add_argument("--binary_output", action="store_true",
                        help="If set, request single-line binary adherence output and enable early stopping.")
    
    # Dataset parameters
    parser.add_argument(
        '--dataset',
        type=str,
        default='all',
        choices=['all'] + possible_datasets,
        help=f"Choose the dataset to use. Options are: all, {', '.join(possible_datasets)}"
    )
    parser.add_argument('--mini', action='store_true', help='Whether to run a mini test.')

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
    model_name = args.model
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
            llm = RaceLLMGGGUF(
                model_dir=model_dir,
                gguf_name=gguf_name,
                chat_format=chat_template,
                binary_output=args.binary_output,
            )
            print(f"Using model {gguf_name} from {model_dir}")
        else:
            if getattr(args, "ax_local", False):
                print("Using local interactive LLM pipeline (no SSH)")
                if "llama-3.2" in args.local_run or "llama-3" in args.local_run:
                    model_name = "local_llama-3"
                elif "phi-3" in args.local_run or "phi3" in args.local_run:
                    model_name = "local_phi-3"
                else:
                    model_name = "local_unknown"
                
                from inference.local_pipeline import LocalLLMPipeline
                llm = LocalLLMPipeline(
                    workdir=args.local_workdir,
                    venv_activate=args.local_venv,
                    run_cmd=args.local_run,
                    verbose=args.local_verbose
                )
            elif getattr(args, "ssh_interactive", False):
                print("Using remote interactive SSH LLM pipeline")
                from inference.remote_pipeline import RemoteLLMPipeline
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
            else:
                print("Using GPU LLM pipeline")
                from inference.inf_pipeline import RaceLLMPipeline
                llm = RaceLLMPipeline(model_dir=model_dir, load_in_4bit=True, chat_template=chat_template, binary_output=args.binary_output) # , max_seq_length=2048, max_new_tokes=2048
            print(f"Using model {args.model} from {model_dir}")

    def _safe_close(obj):
        try:
            close_fn = getattr(obj, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass

    try:
        # Evaluate the decision making on all datasets
        if args.dataset == 'all':
            evaluator = DecisionTester(
                llm=llm,
                model_name=model_name,
                all_tests=True,
                mini=args.mini,
                local=local,
                use_rag=args.rag,
                quant=args.quant,
                rag_offline=args.rag_offline,
                rag_index=args.rag_index,
                rag_corpus=args.rag_corpus,
                rag_max_hits=args.rag_max_hits,
                rag_score_threshold=args.rag_threshold,
                rag_fetch_k=args.rag_fetch_k,
                binary_output=args.binary_output,
            )
            for i, dataset in enumerate(possible_datasets):
                data_dir = os.path.join('tests/decision_tester/robot_states', dataset + '.json')
                # Evaluate the decision making
                evaluator.eval_decision_making(data_dir=data_dir, llm=llm, data_name=dataset)
        # Only evaluate on a specific dataset
        else:
            evaluator = DecisionTester(
                llm=llm,
                model_name=args.model,
                all_tests=False,
                mini=args.mini,
                local=local,
                use_rag=args.rag,
                quant=args.quant,
                rag_offline=args.rag_offline,
                rag_index=args.rag_index,
                rag_corpus=args.rag_corpus,
                rag_max_hits=args.rag_max_hits,
                rag_score_threshold=args.rag_threshold,
                rag_fetch_k=args.rag_fetch_k,
                binary_output=args.binary_output,
            )
            data_dir = os.path.join('tests/decision_tester/robot_states', args.dataset + '.json')
            # Evaluate the decision making
            evaluator.eval_decision_making(data_dir=data_dir, llm=llm, data_name=args.dataset)
    finally:
        _safe_close(llm)
