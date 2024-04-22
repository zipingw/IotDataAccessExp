import numpy as np
import matplotlib.pyplot as plt


def generate_queries(num_queries=100):
    data_min, data_max = 0, 100
    mu = (data_max - data_min) / 2
    sigma = 15
    query_length_mu = 10
    query_length_sigma = 2

    queries = []
    for _ in range(num_queries):
        center = np.random.normal(mu, sigma)
        length = abs(np.random.normal(query_length_mu, query_length_sigma))
        lower_bound = center - length / 2
        upper_bound = center + length / 2
        lower_bound = max(data_min, lower_bound)
        upper_bound = min(data_max, upper_bound)
        queries.append((lower_bound, upper_bound))
    return queries


def plot_queries(queries):
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each query as a line
    for lower_bound, upper_bound in queries:
        ax.plot([lower_bound, upper_bound], [0, 0], marker='|', markersize=10, color='blue', linestyle='None', markeredgewidth=1.5)

    # Customize the plot
    ax.set_title('Visualization of Query Ranges')
    ax.set_yticks([])  # Hide y-axis as it's not meaningful here
    ax.set_xlabel('Data Range')
    plt.xlim([0, 100])
    plt.gca().axes.get_yaxis().set_visible(False)  # Hide the y-axis

    # Show plot
    plt.show()


if __name__ == "__main__":
    queries = generate_queries(100)
    plot_queries(queries)
