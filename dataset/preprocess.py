# import os
# import pandas as pd
# import kagglehub

# # Download/get dataset
# dataset_path = kagglehub.dataset_download("jangedoo/utkface-new")

# print("Dataset path:")
# print(dataset_path)

# records = []

# for root, dirs, files in os.walk(dataset_path):

#     for filename in files:

#         if filename.lower().endswith((".jpg", ".jpeg", ".png")):

#             try:
#                 parts = filename.split("_")

#                 age = int(parts[0])
#                 gender = int(parts[1])
#                 race = int(parts[2])

#                 if gender == 0:
#                     gender_label = "Male"
#                 elif gender == 1:
#                     gender_label = "Female"
#                 else:
#                     continue

#                 image_path = os.path.join(root, filename)

#                 records.append({
#                     "image_path": image_path,
#                     "age": age,
#                     "gender": gender_label
#                 })

#             except Exception:
#                 print("Skipped:", filename)

# df = pd.DataFrame(records)

# print("\nTotal images:", len(df))

# print("\nGender distribution:")
# print(df["gender"].value_counts())

# print("\nAge statistics:")
# print(df["age"].describe())

# df.to_csv(
#     "dataset/utkface_metadata.csv",
#     index=False
# )

# print("\nMetadata saved successfully!")

import os
import pandas as pd
import kagglehub


# ============================================================
# 1. AGE GROUP FUNCTION
# ============================================================

def get_age_group(age):

    if age <= 10:
        return "0-10"

    elif age <= 20:
        return "11-20"

    elif age <= 30:
        return "21-30"

    elif age <= 40:
        return "31-40"

    elif age <= 50:
        return "41-50"

    elif age <= 60:
        return "51-60"

    else:
        return "61+"


# ============================================================
# 2. DOWNLOAD / LOCATE UTKFACE DATASET
# ============================================================

print("Locating UTKFace dataset...")

dataset_path = kagglehub.dataset_download(
    "jangedoo/utkface-new"
)

print("Dataset location:")
print(dataset_path)


# ============================================================
# 3. FIND IMAGE FILES
# ============================================================

image_files = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png"
)

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.lower().endswith(valid_extensions):

            image_files.append(
                os.path.join(root, file)
            )


print()
print("Total image files found:", len(image_files))


# ============================================================
# 4. PROCESS FILENAMES
# ============================================================

records = []

skipped_files = 0


for image_path in image_files:

    filename = os.path.basename(image_path)

    # Example UTKFace filename:
    #
    # 100_0_0_20170119212002382.jpg
    #
    # 100 = age
    # 0   = gender
    # 0   = race
    # remaining = date/time


    try:

        parts = filename.split("_")


        # Need at least:
        # age_gender_race_date

        if len(parts) < 4:
            skipped_files += 1
            continue


        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        age = int(parts[0])


        # ----------------------------------------------------
        # GENDER
        # ----------------------------------------------------

        gender = int(parts[1])


        # UTKFace gender:
        #
        # 0 = Male
        # 1 = Female

        if gender == 0:

            gender_label = "Male"

        elif gender == 1:

            gender_label = "Female"

        else:

            skipped_files += 1
            continue


        # ----------------------------------------------------
        # AGE GROUP
        # ----------------------------------------------------

        age_group = get_age_group(age)


        # ----------------------------------------------------
        # STORE RECORD
        # ----------------------------------------------------

        records.append({

            "image_path": image_path,

            "age": age,

            "age_group": age_group,

            "gender": gender_label

        })


    except (ValueError, IndexError):

        skipped_files += 1

        continue


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# 6. BASIC INFORMATION
# ============================================================

print()
print("=" * 60)

print("DATASET PROCESSING COMPLETED")

print("=" * 60)

print()

print("Valid images:", len(df))

print("Skipped files:", skipped_files)


# ============================================================
# 7. GENDER DISTRIBUTION
# ============================================================

print()
print("Gender distribution:")

print(
    df["gender"].value_counts()
)


# ============================================================
# 8. AGE GROUP DISTRIBUTION
# ============================================================

print()
print("Age group distribution:")

print(
    df["age_group"].value_counts()
)


# ============================================================
# 9. AGE STATISTICS
# ============================================================

print()
print("Age statistics:")

print(
    df["age"].describe()
)


# ============================================================
# 10. SAVE METADATA CSV
# ============================================================

output_file = os.path.join(
    os.path.dirname(__file__),
    "utkface_metadata.csv"
)


df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 11. SUCCESS MESSAGE
# ============================================================

print()

print("=" * 60)

print("Metadata saved successfully!")

print("=" * 60)

print()

print("CSV file:")
print(output_file)

print()

print("Columns:")

print(
    df.columns.tolist()
)

print()

print("First 5 records:")

print(
    df.head()
)