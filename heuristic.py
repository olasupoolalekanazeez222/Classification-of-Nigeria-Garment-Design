import pandas as pd
import itertools

# ======================s
# STEP 0: Load dataset from Excel
# ======================
def load_dataset_from_excel(file_path, sheet_name='sheet1'):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    dataset = {}
    for idx, row in df.iterrows():
        dataset[idx+1] = set(row.dropna().tolist())
    return dataset
# Call the function and assign to dataset
file_path = "dataset2.xlsx"   # Excel file must be in same folder
dataset = load_dataset_from_excel(file_path)

# Example dataset (use this if not loading from Excel)
#dataset = {
    #1: {"A1","B3","C4","D5","E1","F4"},
    #2: {"A2","B3","C4","D1","E2","F3"},
    #3: {"A1","B3","C4","D1","E1","F2"},
    #4: {"A3","B2","C4","D2","E2","F1"},
    #5: {"A4","B1","C2","D3","E2","F3"}
#}

parts = list(dataset.keys())

# ======================
# STEP 1: Compute Rij
# ======================
def compute_Rij(dataset):
    Rij = {}
    for i, j in itertools.permutations(dataset.keys(), 2):
        inter = len(dataset[i].intersection(dataset[j]))
        union = len(dataset[i].union(dataset[j]))
        Rij[(i,j)] = inter/union if union > 0 else 0
    return Rij

Rij = compute_Rij(dataset)

print("\n=== Rij Similarity Matrix ===")
for (i,j), val in Rij.items():
    print(f"R[{i},{j}] = {val:.3f}")

# ======================
# STEP 2: Find minimum Rij -> Indicator group
# ======================
def find_min_Rij(Rij):
    (i,j), min_val = min(Rij.items(), key=lambda x: x[1])
    return {i,j}, (i,j), min_val

indicator_group, min_pair, min_val = find_min_Rij(Rij)
updated_data = set(parts) - indicator_group

print("\n=== Step 2: Initial Indicator Group ===")
print("Indicator Group:", indicator_group)
print("Updated Data:", updated_data)
print("Min Rij pair:", min_pair, "Value:", min_val)

# ======================
# STEP 2 EXTENDED: Expand indicator group until size=N
# ======================
def compute_Yij_Oi(Rij, updated_data, indicator_group, dataset, N):
    while len(indicator_group) < N and updated_data:
        results = {}
        for i in updated_data:
            Oi = {}
            for j in indicator_group:
                remain = set(dataset.keys()) - {i,j}
                k_terms = [Rij[(i,k)] for k in remain if (i,k) in Rij]
                j_terms = [Rij[(j,k)] for k in remain if (j,k) in Rij]
                if remain:
                    avg_term = (sum(k_terms)+sum(j_terms))/(2*len(remain))
                else:
                    avg_term = 0
                val = Rij.get((i,j),0) - avg_term
                Oi[j] = val
                print(f"Y[{i},{j}] = {val:.3f}")
            results[i] = sum(Oi.values())
            print(f"O[{i}] = {results[i]:.3f}")
        best_i = min(results.items(), key=lambda x: x[1])[0]
        indicator_group.add(best_i)
        updated_data.remove(best_i)
        print(f"Added {best_i} to Indicator Group -> {indicator_group}")
    return indicator_group, updated_data

indicator_group, updated_data = compute_Yij_Oi(Rij, updated_data, indicator_group, dataset, N=4)

# ======================
# STEP 3: Matching updateddata to groups
# ======================
def step3_matching(updated_data, indicator_group, dataset, Rij):
    # split group elements into singletons
    groups = [{i} for i in indicator_group]

    while updated_data:
        results = {}
        for i in list(updated_data):
            for g in groups:
                remain = set(dataset.keys()) - g - {i}
                k_terms = [Rij.get((i,k),0) for k in g]
                avg_g = sum(k_terms)/len(g) if g else 0
                if remain:
                    avg_remain = sum(Rij.get((i,l),0) for l in remain)/len(remain)
                else:
                    avg_remain = 0
                Mij = avg_g - avg_remain
                results[(i,tuple(g))] = Mij
                print(f"M[{i},{g}] = {Mij:.3f}")
        best_pair = max(results.items(), key=lambda x: x[1])[0]
        i, g_tuple = best_pair
        for g in groups:
            if set(g) == set(g_tuple):
                g.add(i)
        updated_data.remove(i)
        print(f"Assigned {i} -> {g_tuple}")
    return groups

groups = step3_matching(updated_data, indicator_group, dataset, Rij)

print("\n=== Final Groups after Step 3 ===")
print(groups)

# ======================
# STEP 4: Objective Function H
# ======================
def compute_objective_H(groups, Rij):
    H_values = {}
    for idx, group in enumerate(groups, start=1):
        if len(group) <= 1:
            H_values[idx] = 0
            continue
        pairs = [(p,k) for p in group for k in group if p != k]
        num_pairs = len(pairs)
        sum_R = sum(Rij.get((p,k),0) for (p,k) in pairs)
        H_values[idx] = sum_R / num_pairs if num_pairs else 0

        # print Rpk table for this group
        print(f"\nRpk Matrix for Group {idx}: {group}")
        for p in group:
            row = []
            for k in group:
                row.append(f"{Rij.get((p,k),0):.2f}" if p!=k else "-")
            print(f"{p}: {row}")

    H_total = sum(H_values.values()) / len(groups) if groups else 0
    return H_values, H_total

H_values, H_total = compute_objective_H(groups, Rij)

print("\n=== Step 4 Objective Function ===")
for j,h in H_values.items():
    print(f"H for Group {j}: {h:.3f}")
print(f"Overall H = {H_total:.3f}")