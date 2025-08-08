import os, re, json
from collections import defaultdict
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

slide_type = "PAP"
# slide_names = [os.path.basename(name) for name in glob(f"/vol0/nfs9/tileimage/new/1024/*/{slide_type}/*")]
slide_names = [os.path.basename(name) for name in glob(f"/vol0/nfs9/tileimage/250529_inference/1024/*/{slide_type}/*")]

json_data_path = "position.json"
with open(json_data_path, 'r', encoding='utf-8-sig') as json_file: data = json.load(json_file)

# Initialize the result dictionary
result = defaultdict(lambda: defaultdict(int))
diagnosis_counts = defaultdict(int)

skipped = 0
# Process the data
for entry in data:
    slide_code = entry['SlideCode']
    if slide_code not in slide_names: skipped+=1; continue
    for pos in entry['Position']:
        diagnosis = pos['DiagnosisName']
        result[slide_code][diagnosis] += 1
        diagnosis_counts[diagnosis] += 1

# If you want to convert the result to a normal dict (optional)
result = {slide: dict(diags) for slide, diags in result.items()}

print(diagnosis_counts)
acronyms = []
for name in diagnosis_counts.keys():
    found = re.findall(r'\((.*?)\)', name)
    if found:
        acronyms.append(found[0])
    else:
        acronyms.append(name)  # or '' if you prefer an empty string


d_names = [a for a in acronyms if a is not None]

# d_names = [re.findall(r'\((.*?)\)', name)[0] for name in diagnosis_counts.keys()]
d_counts = list(diagnosis_counts.values())

# Build DataFrame
df = pd.DataFrame({'Diagnosis': d_names, 'Count': d_counts}).sort_values('Count', ascending=False)

# Set up style
sns.set_theme(style="whitegrid")
sns.set_palette("Spectral")
plt.rcParams['font.size'] = 12

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot vertical bar chart
bars = sns.barplot(x='Diagnosis', y='Count', data=df, 
                  edgecolor='darkgray', linewidth=0.7, ax=ax)

# Customize elements

plt.title(f'{slide_type} Cervical Cytology Diagnosis Distribution', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Number of Cases', labelpad=15, fontsize=14)
plt.xlabel('Diagnosis Name', labelpad=15, fontsize=14)
plt.xticks(fontsize=12, rotation=0, ha='center')  # 약어가 한글로 바뀐다면 ha='right' 권장
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for p in bars.patches:
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2, height + max(df['Count']) * 0.01,
             f'{int(height):,}', 
             va='bottom', ha='center', fontsize=12)

sns.despine(left=True, bottom=True)

plt.text(0.5, -0.18, 'Data Source: Cervical Cancer Screening Project | Visualization: Rinorbit AI Team',
         transform=ax.transAxes, ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(f'{slide_type}_diagnosis_distribution_vertical.png', dpi=300, bbox_inches='tight')
plt.show()