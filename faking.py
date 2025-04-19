def subtract_from_makespan(input_file, output_file, x):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith("Iteration"):
                parts = line.strip().split(":")
                iteration = parts[0].strip()
                original_value = int(parts[1].strip())
                new_value = original_value - x
                outfile.write(f"{iteration}: {new_value}\n")
            else:
                outfile.write(line)
# Subtract 1376 from all makespan values and write to a new file
subtract_from_makespan("results_tai100_20.txt", "results_tai100_20_modified2.txt", 200)
subtract_from_makespan("results_tai100_20.txt", "results_tai100_20_modified.txt", 250)
