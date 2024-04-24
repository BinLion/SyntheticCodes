
from codeT.execution import best_solution, pass_most_solution
from dataset_gen.filtering import load_solutions_with_tests


# data = load_solutions_with_tests('/home/ec2-user/SyntheticCodes/data/40B_temp1_50k_v1/dataset_exercise_40B_temp1_50k_solutions_withtests_v1.jsonl')
data = load_solutions_with_tests('/home/ec2-user/SyntheticCodes/test2.jsonl')
dataset = []
for item in data:
    result = best_solution(item, 1, 5, True)
    print(f"{item['exercise_id']} 1\n")
    best = pass_most_solution(result)
    print(f"{item['exercise_id']} 2\n")
    print(result, best)
    dataset.append({"exercise_id": best['task_id'], "problem": best['prompt'], "solution": best['completion']})
