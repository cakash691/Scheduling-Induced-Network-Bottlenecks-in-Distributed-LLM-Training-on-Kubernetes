# G23_plot.py
import matplotlib.pyplot as plt
import matplotlib

# Set font to Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'


# STRICTLY HARDCODED VALUES - DO NOT IMPORT CSV HERE
configurations = ['Pod Affinity\n(Same Node)', 'Pod Anti-Affinity\n(Different Nodes)']
execution_times_seconds = [11.98, 14.74] 

plt.figure(figsize=(8, 6))
bars = plt.bar(configurations, execution_times_seconds, color=['#4CAF50', '#F44336'])

# Adding value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval}s", ha='center', va='bottom', fontweight='bold')

plt.title('G23: Distributed ELECTRA Training Execution Time', fontweight='bold')
plt.ylabel('Execution Time (Seconds)', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('G23_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as G23_plot.png")
