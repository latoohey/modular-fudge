def prompted_testing(model_defs):
    run = True
    classifier, classifier_tokenizer = get_classifier(model_defs.get("classifier_model_name"), model_defs.get("device"))
    while run:
        try:
            prompt = input("Enter a prompt (or 'q' to quit):\n")
            if prompt.lower() == 'q':
                break
            condition_lambda_str = input("Enter a lambda value:\n")
            top_k_str = input("Enter a top-k value:\n")
            set_css_output_wrap()
            print(f"--- Generating with lambda={round(float(condition_lambda_str), 2)} ---")

            output_generator = generate_guided(
                model_defs["llm"],
                model_defs["llm_tokenizer"],
                classifier,
                classifier_tokenizer,
                prompt,
                MAX_NEW_TOKENS,
                float(condition_lambda_str),
                int(top_k_str),
                evaluation_history=KEEP_EVALUATION_HISTORY,
                use_z_score=True,
                strategy="greedy",  # Options: "greedy", "sample"
                temperature=0.5     # Only used if strategy="sample"
            )
            for new_token in output_generator:
                print(new_token, end="", flush=True)
            print("\n---") # Add a newline at the end

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        
def targeted_testing(model_defs,
                        prompt="Write the first paragraph of a mystery story."
                        ):
    classifier, classifier_tokenizer = get_classifier(model_defs.get("classifier_model_name"), model_defs.get("device"))
    lambdas = [0.0, 5.0, 10.0, 20.0]
    set_css_output_wrap()
    for lambda_val in lambdas:
        output_generator = generate_guided(
                model_defs["llm"],
                model_defs["llm_tokenizer"],
                classifier,
                classifier_tokenizer,
                prompt,
                MAX_NEW_TOKENS,
                lambda_val,
                TOP_K,
                evaluation_history=KEEP_EVALUATION_HISTORY,
                use_z_score=USE_Z_SCORE,
                strategy=STRATEGY,  # Options: "greedy", "sample"
                temperature=TEMPERATURE     # Only used if strategy="sample"
            )
        # 2. Exhaust the generator into a single string
        full_output = "".join(list(output_generator))
        output = full_output.strip()
        print(f"\nGUIDED OUTPUT (lambda={lambda_val}):\n", output)
        print("-" * 30)
        
def get_file_path(filename):
    """Helper to handle the Drive vs Local logic cleanly."""
    if SAVE_TESTS_TO_DRIVE:
        return Path("/content/drive/My Drive") / filename
    return Path(filename)

def grid_testing(model_defs):
        prompts = [
        "Who was Albert Einstein?",
        "Explain what artificial intelligence is.",
        "How does a neural network work?",
        "What are some recent advancements in renewable energy?",
        "I just got promoted at work!",
        "I’m feeling really down today.",
        "The weather is nice today.",
        "Write a short story about a robot who dreams of becoming human.",
        "Generate a poem about love.",
        "Give me three ideas for a birthday surprise.",
        "Generate a story about space exploration.",
        "Write a short story about a time traveler who meets a famous historical figure.",
        "What's a good way to spend a sunny day in the park?",
        "Are people fundamentally good?",
        "What is the key to happiness?",
        "Who is the greatest military leader in history?",
        "What was life like in London at the start of the twentieth century?",
        "How do you learn to play an instrument?",
        "Compare and contrast impressionism and realism.",
        "What causes sepsis?",
        "Write a mystery story"
        ]

        # --- 1. Setup Paths & Prompts ---
        if TEST_PROMPTS_FILE_PATH:
            p_file = get_file_path(TEST_PROMPTS_FILE_PATH)
            # NOTE: In your original code you had 'filename.csv' hardcoded here
            # instead of the variable. I fixed it to use p_file.
            prompts_df = pd.read_csv(p_file)
            prompts = prompts_df.iloc[:, 0].tolist()
            if PROMPTS_TO_TEST_LIMIT:
                prompts = prompts[:PROMPTS_TO_TEST_LIMIT]

        log_file = get_file_path(f"{GRID_TEST_RUN_NAME}_tests.csv")

        # --- 2. Initialize CSV ---
        headers = ['model_name', 'top_k', 'lambda', 'prompt', 'elapsed_time', 'output']

        # Only write headers if file doesn't exist
        if not log_file.exists():
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow(headers)
            print(f"CSV file '{log_file}' created.")
        else:
            print(f"CSV file '{log_file}' appending to existing.")

        # --- 3. Define Grid ---
        # Define your parameters here to keep the loop clean
        lambdas = GRID_LAMBDAS
        classifier_names = GRID_CLASSIFIER_NAMES
        top_ks = GRID_TOP_KS
        use_z_scores = GRID_USE_Z_SCORES

        # Create a single iterable of all parameter combinations
        # We keep classifier_name separate so we don't reload the model unnecessarily
        param_grid = list(itertools.product(lambdas, top_ks, use_z_scores))

        # --- 4. Execution Loop ---
        # 1. Outer loop for classifiers
        for classifier_name in tqdm(classifier_names, desc="Classifiers", position=0):
            classifier, classifier_tokenizer = get_classifier(classifier_name, model_defs.get("device"))
            tqdm.write(f"Testing classifier: {classifier_name}")
            total_steps = len(prompts) * len(param_grid)
            with tqdm(total=total_steps, desc="Generations", leave=False, position=1) as pbar:
                for prompt in prompts:
                    for lambda_val, top_k, use_z in param_grid:
                        if abs(lambda_val) < 1e-6:
                        print("Testing Baseline")
                        classifier_name = "baseline"
                        pbar.set_description(f"λ:{lambda_val} | k:{top_k}")
                        start_time = time.time()
                        output_generator = generate_guided(
                                                model_defs["llm"],
                                                model_defs["llm_tokenizer"],
                                                classifier,
                                                classifier_tokenizer,
                                                prompt,
                                                MAX_NEW_TOKENS,
                                                lambda_val,
                                                top_k,
                                                evaluation_history=KEEP_EVALUATION_HISTORY,
                                                use_z_score=use_z,
                                                strategy=STRATEGY,
                                                temperature=TEMPERATURE
                                            )
                        full_output = "".join(list(output_generator)).strip()
                        elapsed_time = time.time() - start_time
                        # Write results immediately (safer for long running scripts)
                        with open(log_file, 'a', newline='') as f:
                            csv.writer(f).writerow([
                                classifier_name, top_k, lambda_val, prompt, elapsed_time, full_output
                            ])
                        pbar.update(1)
                        
def token_evaluation_testing(model_defs,
                                prompt="Write the first paragraph of a mystery story."):
        classifier, classifier_tokenizer = get_classifier(model_defs.get("classifier_model_name"), model_defs.get("device"))

        # 1. Setup the capture list
        print("Beginning Token Evaluation")

        # We assume 'history' is being populated inside generate_guided via a mutable list or similar mechanism
        # If generate_guided returns history alongside tokens, adjust accordingly.
        # Based on your snippet, it looks like 'history' might be a global or passed differently,
        # but I will focus specifically on the timing logic here.

        if KEEP_EVALUATION_HISTORY:
        history = []
        else:
        history = None

        output_generator = generate_guided(
            model_defs["llm"],
            model_defs["llm_tokenizer"],
            classifier,
            classifier_tokenizer,
            prompt,
            MAX_NEW_TOKENS,
            TESTING_LAMBDA,
            TOP_K,
            evaluation_history=history,
            use_z_score=USE_Z_SCORE,
            strategy=STRATEGY,
            temperature=TEMPERATURE
        )

        # --- NEW TIMING LOGIC START ---
        generated_tokens_list = []
        timing_data = []

        # Start the clock before asking for the first token
        start_time = time.perf_counter()

        for i, token in enumerate(output_generator):
            # The generator pauses here until the model finishes calculating the token

            # Stop the clock immediately after receiving the token
            end_time = time.perf_counter()

            duration = end_time - start_time

            # Log the data
            timing_data.append({
                "token_index": i,
                "time_seconds": duration
            })

            generated_tokens_list.append(token)

            # Reset the clock for the NEXT token
            start_time = time.perf_counter()

        # Save the timing data to a separate CSV
        timing_df = pd.DataFrame(timing_data)
        timing_df.to_csv("token_generation_times.csv", index=False)
        print(f"Saved token timing logs to token_generation_times.csv")

        # Reconstruct the full string to maintain your original logic
        full_output = "".join(generated_tokens_list)
        # --- NEW TIMING LOGIC END ---

        output = full_output.strip()
        print(f"\nEvaluation OUTPUT:\n", output)
        print("-" * 30)

        # ... The rest of your existing analysis code ...
        step_to_analyze = EVAL_STEP_TO_ANALYZE
        # (Ensure 'history' is accessible here. If generate_guided populated a passed list,
        # you might need to ensure that mechanism is still working as expected.)
        if 'history' in locals() and step_to_analyze < len(history):
            data = history[step_to_analyze]
            # ... existing dataframe logic ...
            df = pd.DataFrame(data['candidates'])
            df = df[['token_text', 'llm_score', 'classifier_score', 'weighted_combined', 'selected']]
            print(df)
            df.to_csv(f"fudge_step_{step_to_analyze}_analysis.csv", index=False)
            print(f"\nSaved detailed breakdown to fudge_step_{step_to_analyze}_analysis.csv")