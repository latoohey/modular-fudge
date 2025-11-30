# ==========================================
# 1. SETUP: Maps and Styles
# ==========================================

# Bridge the simple CSV filenames to your internal keys
csv_name_map = {
    'llama': 'llama',
    'lstm': 'lstm_2_256',
    'mamba': 'mamba_128_4_16_1'
}

MY_LABEL_MAP = {
    'lstm_2_256': 'LSTM',
    'mamba_128_4_16_1': 'Mamba',
    'llama': 'Llama 3B Instruct'
}

MY_COLOR_MAP = {
    'LSTM': '#3498db',             # Blue
    'Mamba': '#e67e22',            # Orange
    'Llama 3B Instruct': '#95a5a6' # Grey
}

# ==========================================
# 2. LOAD DATA
# ==========================================
data_frames = []
file_pattern = re.compile(r'token_generation_times_([a-zA-Z]+)_(\d+)\.csv')

for filename in os.listdir('.'):
    match = file_pattern.match(filename)
    if match:
        raw_model_name = match.group(1) # e.g., 'lstm'
        df = pd.read_csv(filename)
        df['model'] = raw_model_name
        data_frames.append(df)

all_data = pd.concat(data_frames, ignore_index=True)

# Average the tests
averaged_data = all_data.groupby(['model', 'token_index'])['time_seconds'].mean().reset_index()
pivot_data = averaged_data.pivot(index='token_index', columns='model', values='time_seconds')

# ==========================================
# 3. RENAME COLUMNS TO FINAL DISPLAY NAMES
# ==========================================
# This maps 'lstm' -> 'lstm_2_256' -> 'LSTM'
rename_dict = {
    col: MY_LABEL_MAP[csv_name_map[col]]
    for col in pivot_data.columns
    if col in csv_name_map
}
pivot_data.rename(columns=rename_dict, inplace=True)

# Slice the specific range (10 to 300)
plot_data = pivot_data.loc[10:300]

# ==========================================
# 4. PLOTTING
# ==========================================
baseline_name = 'Llama 3B Instruct'

# --- Graph 1: Total Generation Time ---
plt.figure(figsize=(10, 6))

for model_name in plot_data.columns:
    plt.plot(
        plot_data.index,
        plot_data[model_name],
        label=model_name,
        color=MY_COLOR_MAP[model_name],
        linewidth=2
    )

plt.title('Average Token Generation Time per Model (Tokens 10-300)')
plt.xlabel('Token Index')
plt.ylabel('Time (seconds)')
plt.xlim(10, 300)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('graph1_styled.png')
plt.show()

# --- Graph 2: Baseline vs Added Time ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Baseline (Llama 3B Instruct)
ax1.plot(
    plot_data.index,
    plot_data[baseline_name],
    color=MY_COLOR_MAP[baseline_name],
    label=baseline_name,
    linewidth=2
)
ax1.set_title(f'Baseline Performance ({baseline_name})')
ax1.set_xlabel('Token Index')
ax1.set_ylabel('Time (seconds)')
ax1.set_xlim(10, 300)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Added Time (The Deltas)
# Dynamically find models that are NOT the baseline
comparison_models = [m for m in plot_data.columns if m != baseline_name]

for model in comparison_models:
    # Calculate difference
    delta = plot_data[model] - plot_data[baseline_name]

    ax2.plot(
        plot_data.index,
        delta,
        color=MY_COLOR_MAP[model],
        label=f'{model} (Added Cost)',
        linewidth=2
    )

ax2.set_title('Additional Time Cost vs Baseline')
ax2.set_xlabel('Token Index')
ax2.set_ylabel('Added Time (seconds)')
ax2.set_xlim(10, 300)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graph2_styled_deltas.png')
plt.show()