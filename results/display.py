import matplotlib.pyplot as plt
import csv

# Separate files into two groups: with "-cl" and others
non_cl_files = ['speedup1.csv', 'speedup2.csv', 'speedup3.csv']
cl_files = ['speedup1-cl.csv', 'speedup2-cl.csv']

try:
    x_non_cl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # X-axis for non "-cl" files (1 to 12)
    x_cl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # X-axis for "-cl" files (1 to 20)
    all_data_cl = {}  # Dictionary to store data from "-cl" files
    all_data_non_cl = {}  # Dictionary to store data from non "-cl" files

    # Process "-cl" files
    for filename in cl_files:
        y1 = []  # First y-axis data
        y2 = []  # Second y-axis data
        with open(filename, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)  # Skip the header row if present

            # Extract data
            for row in csvreader:
                y1.append(float(row[1]))  # Second column as first y-axis
                y2.append(float(row[2]))  # Third column as second y-axis

        # Store data for the current "-cl" file
        all_data_cl[filename] = (y1, y2)

    # Process non "-cl" files
    for filename in non_cl_files:
        y1 = []  # First y-axis data
        y2 = []  # Second y-axis data
        with open(filename, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)  # Skip the header row if present

            # Extract data
            for row in csvreader:
                y1.append(float(row[1]))  # Second column as first y-axis
                y2.append(float(row[2]))  # Third column as second y-axis

        # Store data for the current non "-cl" file
        all_data_non_cl[filename] = (y1, y2)

    # Plot for non "-cl" files (Execution Times)
    plt.figure()
    for filename, (y1, _) in all_data_non_cl.items():
        plt.plot(x_non_cl, y1, marker='x', label=f'{filename}')
    plt.title('Execution Times - Local')
    plt.xlabel('# threads')
    plt.ylabel('time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot for non "-cl" files (Speedups) with diagonal line
    plt.figure()
    for filename, (_, y2) in all_data_non_cl.items():
        plt.plot(x_non_cl, y2, marker='x', label=f'{filename}')
    max_value = max(x_non_cl)  # Determine the range for the diagonal line
    plt.plot(range(1, max_value + 1), range(1, max_value + 1), linestyle='--', color='black')
    plt.title('Speedups - Local')
    plt.xlabel('# threads')
    plt.ylabel('speedup')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot for "-cl" files (Execution Times)
    plt.figure()
    for filename, (y1, _) in all_data_cl.items():
        plt.plot(x_cl, y1, marker='x', label=f'{filename}')
    plt.title('Execution Times - Cluster')
    plt.xlabel('# threads')
    plt.ylabel('time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot for "-cl" files (Speedups) with diagonal line
    plt.figure()
    for filename, (_, y2) in all_data_cl.items():
        plt.plot(x_cl, y2, marker='x', label=f'{filename}')
    max_value = max(x_cl)  # Determine the range for the diagonal line
    plt.plot(range(1, max_value + 1), range(1, max_value + 1), linestyle='--', color='black')
    plt.title('Speedups - Cluster')
    plt.xlabel('# threads')
    plt.ylabel('speedup')
    plt.grid(True)
    plt.legend()
    plt.show()

except Exception as e:
    print(f"Error: {e}")
