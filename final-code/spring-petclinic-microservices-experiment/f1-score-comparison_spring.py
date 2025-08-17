import matplotlib.pyplot as plt

# Data for the bar chart
categories = ['Pre-Fine-Tuning', 'Post-Fine-Tuning']
f1_scores = [0.70, 0.735]

# Create the bar chart
plt.figure(figsize=(8, 6))
plt.bar(categories, f1_scores, color=['blue', 'green'], width=0.4)

# Add labels and title
plt.xlabel('Model Stage')
plt.ylabel('F1-Score')
plt.title('Comparison of F1-Scores Before and After Fine-Tuning on Spring PetClinic Benchmark')
plt.ylim(0, 1)  # Set y-axis limit for better visualization

# Add value labels on top of the bars
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.01, str(v), ha='center', fontweight='bold')

# Save the figure to a file (you can include this image in your LaTeX report)
plt.savefig('f1_score_comparison_spring.png')

# Optionally display the plot
plt.show()