import random

def random_lines(input_file, output_file, N):
    reservoir = []
    with open(input_file, 'r') as file:
        # print(f"total lines: {len(file.readlines())}")
        # Fill the reservoir with first N lines
        for _ in range(N):
            line = file.readline()
            if not line:  # 文件行数少于N
                return
            reservoir.append(line)
        
        # Line number, starting from N+1
        line_num = N + 1
        for line in file:
            line_num += 1
            # Pick a random integer between 0 and line_num-1, inclusive
            rand = random.randrange(line_num)
            if rand < N:
                reservoir[rand] = line
        
    # Write the chosen lines to the output file
    with open(output_file, 'w') as file:
        file.writelines(reservoir)
        print(f"new lines: {len(reservoir)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="shorten files under dataset to speed up the training process")
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--lines", type=int, required=1000)

    args = parser.parse_args()
    print(args)

    # 使用函数
    random_lines(args.src, args.target, args.lines)
    # random_lines('dataset/pretrain_hq.jsonl', 'dataset/pretrain_hq_mini.jsonl', lines)
    # random_lines('dataset/sft_mini_512.jsonl', 'dataset/sft_mini_512_mini.jsonl', lines)

