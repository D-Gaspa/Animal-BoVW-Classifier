import matplotlib.pyplot as plt

filters = [
    "No Filtering",
    "Gabor",
    "Histogram Equalization",
    "Histogram Equalization + Gabor",
    "Canny Edge Detector",
    "Unsharp Mask Filter",
    "Enhancement + Background Removal",
    "Background Removal",
    "Background Removal + Gabor",
    "Background Removal + Histogram Equalization",
    "Background Removal + Unsharp Mask Filter",
    "Background Removal + Canny Edge Detector",
    "Background Removal + Histogram Equalization + Unsharp Mask Filter",
    "Background Removal + Unsharp Mask Filter + Histogram Equalization",
    "Background Removal + Histogram Equalization + Unsharp Mask Filter + Gabor"
]

accuracies = [
    0.67,
    0.59,
    0.65,
    0.58,
    0.45,
    0.68,
    0.78,
    0.82,
    0.65,
    0.83,
    0.84,
    0.63,
    0.80,
    0.78,
    0.80
]

# Sort the filters based on accuracies
sorted_indices = sorted(range(len(accuracies)), key=lambda k: accuracies[k])
filters = [filters[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]

# Creating bar chart
plt.figure(figsize=(10, 8))
plt.barh(filters, accuracies, color='skyblue')
plt.xlabel('Accuracy')
plt.ylabel('Filter Combinations')
plt.title('Accuracy of Different Filter Combinations')

# Highlight the bar with the highest accuracy
plt.barh(filters[-1], accuracies[-1], color='green')

# Adding the accuracy values on the bars for clarity
for i in range(len(filters)):
    plt.text(accuracies[i], i, f'{accuracies[i]:.2f}', va='center', color='blue' if i < len(filters) - 1 else 'white',
             fontweight='bold')

# Save the plot to a file
plt.savefig('filter_combination_accuracies.png', bbox_inches='tight')

# Show the plot
plt.show()
