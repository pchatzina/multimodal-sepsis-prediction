-- ------------------------------------------------------------------
-- Step 0: Set Seed for Reproducibility
-- This ensures the random sort order is the same every time you run this.
-- ------------------------------------------------------------------
SELECT setseed(0.2024);

-- ------------------------------------------------------------------
-- Step 1: Define the "Modality Signature" for every patient
-- ------------------------------------------------------------------
DROP TABLE IF EXISTS mimiciv_ext.patient_strata;
CREATE TABLE mimiciv_ext.patient_strata AS
SELECT 
    c.subject_id,
    c.hadm_id,
    c.sepsis_label,
    -- Create a text signature: e.g., 'EHR_CXR' or 'EHR_ECG_CXR'
    CONCAT(
        'EHR',
        CASE WHEN cxr.hadm_id IS NOT NULL THEN '_CXR' ELSE '' END,
        CASE WHEN ecg.hadm_id IS NOT NULL THEN '_ECG' ELSE '' END
    ) AS modality_signature
FROM mimiciv_ext.cohort c
LEFT JOIN mimiciv_ext.cohort_cxr cxr 
    ON c.subject_id = cxr.subject_id AND c.hadm_id = cxr.hadm_id
LEFT JOIN mimiciv_ext.cohort_ecg ecg 
    ON c.subject_id = ecg.subject_id AND c.hadm_id = ecg.hadm_id;

-- ------------------------------------------------------------------
-- Step 2: Assign Split Labels (Train/Val/Test)
-- Stratified by BOTH Sepsis Label AND Modality Signature
-- ------------------------------------------------------------------
DROP TABLE IF EXISTS mimiciv_ext.dataset_splits;
CREATE TABLE mimiciv_ext.dataset_splits AS
WITH randomized AS (
    SELECT 
        subject_id,
        hadm_id,
        modality_signature,
        sepsis_label,
        -- Assign a random number within each unique group (Stratum)
        ROW_NUMBER() OVER (
            PARTITION BY modality_signature, sepsis_label 
            ORDER BY RANDOM()
        ) as rn,
        COUNT(*) OVER (
            PARTITION BY modality_signature, sepsis_label
        ) as total_in_group
    FROM mimiciv_ext.patient_strata
)
SELECT 
    subject_id,
    hadm_id,
    modality_signature,
    sepsis_label,
    CASE 
        -- 70% Training
        WHEN rn <= (total_in_group * 0.70) THEN 'train'
        -- 15% Validation (next 15%)
        WHEN rn <= (total_in_group * 0.85) THEN 'validate'
        -- 15% Test (remainder)
        ELSE 'test'
    END AS dataset_split
FROM randomized;

-- ------------------------------------------------------------------
-- Step 3: Verification / Sanity Check
-- Shows the count of patients in each split/modality combo
-- ------------------------------------------------------------------
SELECT 
    dataset_split,
    modality_signature,
    COUNT(*) as num_patients,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY modality_signature), 1) as percentage
FROM mimiciv_ext.dataset_splits
GROUP BY dataset_split, modality_signature
ORDER BY modality_signature, dataset_split;