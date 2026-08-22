-- Collapse raw to 3 datasets and hang catalog.course on raw.dataset
-- so the ERD is one chain: raw → identity → prediction → recommendation.

SET statement_timeout = 0;
SET maintenance_work_mem = '1GB';

INSERT INTO raw.dataset (dataset_key, display_name) VALUES
    ('student_mat', 'UCI Student Performance (Math)'),
    ('student_por', 'UCI Student Performance (Portuguese)'),
    ('oulad', 'Open University Learning Analytics Dataset')
ON CONFLICT (dataset_key) DO NOTHING;

DO $$
BEGIN
    IF to_regclass('raw.uci_mat') IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM raw.student_mat LIMIT 1) THEN
        INSERT INTO raw.student_mat (
            school, sex, age, address, famsize, pstatus, medu, fedu, mjob, fjob,
            reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid,
            activities, nursery, higher, internet, romantic, famrel, freetime, goout,
            dalc, walc, health, absences, g1, g2, g3
        )
        SELECT
            payload->>'school', payload->>'sex', (payload->>'age')::int,
            payload->>'address', payload->>'famsize', payload->>'Pstatus',
            (payload->>'Medu')::int, (payload->>'Fedu')::int,
            payload->>'Mjob', payload->>'Fjob', payload->>'reason', payload->>'guardian',
            (payload->>'traveltime')::int, (payload->>'studytime')::int,
            (payload->>'failures')::int, payload->>'schoolsup', payload->>'famsup',
            payload->>'paid', payload->>'activities', payload->>'nursery',
            payload->>'higher', payload->>'internet', payload->>'romantic',
            (payload->>'famrel')::int, (payload->>'freetime')::int, (payload->>'goout')::int,
            (payload->>'Dalc')::int, (payload->>'Walc')::int, (payload->>'health')::int,
            (payload->>'absences')::int, (payload->>'G1')::int, (payload->>'G2')::int,
            (payload->>'G3')::int
        FROM raw.uci_mat
        ORDER BY row_id;
    END IF;

    IF to_regclass('raw.uci_por') IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM raw.student_por LIMIT 1) THEN
        INSERT INTO raw.student_por (
            school, sex, age, address, famsize, pstatus, medu, fedu, mjob, fjob,
            reason, guardian, traveltime, studytime, failures, schoolsup, famsup, paid,
            activities, nursery, higher, internet, romantic, famrel, freetime, goout,
            dalc, walc, health, absences, g1, g2, g3
        )
        SELECT
            payload->>'school', payload->>'sex', (payload->>'age')::int,
            payload->>'address', payload->>'famsize', payload->>'Pstatus',
            (payload->>'Medu')::int, (payload->>'Fedu')::int,
            payload->>'Mjob', payload->>'Fjob', payload->>'reason', payload->>'guardian',
            (payload->>'traveltime')::int, (payload->>'studytime')::int,
            (payload->>'failures')::int, payload->>'schoolsup', payload->>'famsup',
            payload->>'paid', payload->>'activities', payload->>'nursery',
            payload->>'higher', payload->>'internet', payload->>'romantic',
            (payload->>'famrel')::int, (payload->>'freetime')::int, (payload->>'goout')::int,
            (payload->>'Dalc')::int, (payload->>'Walc')::int, (payload->>'health')::int,
            (payload->>'absences')::int, (payload->>'G1')::int, (payload->>'G2')::int,
            (payload->>'G3')::int
        FROM raw.uci_por
        ORDER BY row_id;
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('raw.oulad_courses') IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM raw.oulad LIMIT 1) THEN
        INSERT INTO raw.oulad (source_file, payload)
        SELECT 'courses.csv', to_jsonb(s) - 'loaded_at'
        FROM raw.oulad_courses s;

        INSERT INTO raw.oulad (source_file, payload)
        SELECT 'assessments.csv', to_jsonb(s) - 'loaded_at'
        FROM raw.oulad_assessments s;

        INSERT INTO raw.oulad (source_file, payload)
        SELECT 'vle.csv', to_jsonb(s) - 'loaded_at'
        FROM raw.oulad_vle s;

        INSERT INTO raw.oulad (source_file, payload)
        SELECT 'studentInfo.csv', to_jsonb(s) - 'loaded_at'
        FROM raw.oulad_student_info s;

        INSERT INTO raw.oulad (source_file, payload)
        SELECT 'studentRegistration.csv', to_jsonb(s) - 'loaded_at'
        FROM raw.oulad_student_registration s;

        INSERT INTO raw.oulad (source_file, payload)
        SELECT 'studentAssessment.csv', to_jsonb(s) - 'loaded_at'
        FROM raw.oulad_student_assessment s;

        INSERT INTO raw.oulad (source_file, payload)
        SELECT 'studentVle.csv', to_jsonb(s) - 'loaded_at' - 'row_id'
        FROM raw.oulad_student_vle s;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'catalog' AND table_name = 'course' AND column_name = 'dataset_key'
    ) THEN
        ALTER TABLE catalog.course ADD COLUMN dataset_key TEXT;
    END IF;
END $$;

UPDATE catalog.course
SET dataset_key = CASE
    WHEN course_code = 'math' THEN 'student_mat'
    WHEN course_code = 'portuguese' THEN 'student_por'
    ELSE 'oulad'
END
WHERE dataset_key IS NULL;

ALTER TABLE catalog.course ALTER COLUMN dataset_key SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'course_dataset_fkey') THEN
        ALTER TABLE catalog.course
            ADD CONSTRAINT course_dataset_fkey
            FOREIGN KEY (dataset_key) REFERENCES raw.dataset (dataset_key);
    END IF;
END $$;

UPDATE raw.dataset d SET files = COALESCE(src.files, '[]'::jsonb)
FROM (
    SELECT 'student_mat'::text AS dataset_key,
           jsonb_build_array(jsonb_build_object('file','student-mat.csv','rows', COUNT(*))) AS files
    FROM raw.student_mat
    UNION ALL
    SELECT 'student_por',
           jsonb_build_array(jsonb_build_object('file','student-por.csv','rows', COUNT(*)))
    FROM raw.student_por
    UNION ALL
    SELECT 'oulad',
           COALESCE(jsonb_agg(jsonb_build_object('file', source_file, 'rows', n) ORDER BY source_file), '[]'::jsonb)
    FROM (SELECT source_file, COUNT(*) AS n FROM raw.oulad GROUP BY source_file) t
) src
WHERE d.dataset_key = src.dataset_key;

DROP TABLE IF EXISTS raw.uci_mat;
DROP TABLE IF EXISTS raw.uci_por;
DROP TABLE IF EXISTS raw.load_manifest;
DROP TABLE IF EXISTS raw.oulad_student_vle;
DROP TABLE IF EXISTS raw.oulad_student_assessment;
DROP TABLE IF EXISTS raw.oulad_student_registration;
DROP TABLE IF EXISTS raw.oulad_student_info;
DROP TABLE IF EXISTS raw.oulad_assessments;
DROP TABLE IF EXISTS raw.oulad_vle;
DROP TABLE IF EXISTS raw.oulad_courses;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'catalog' AND table_name = 'student' AND column_name = 'full_name'
    ) THEN
        EXECUTE $q$ALTER TABLE catalog.student DROP COLUMN IF EXISTS full_name$q$;
        EXECUTE $q$ALTER TABLE catalog.student DROP COLUMN IF EXISTS email$q$;
    END IF;
END $$;

COMMENT ON SCHEMA raw IS 'Three source datasets: student-mat, student-por, OULAD.';
COMMENT ON SCHEMA catalog IS 'Identity: one student–course enrollment, fed by raw.dataset.';
COMMENT ON SCHEMA prediction IS 'Hybrid CNN-BiLSTM risk per enrollment and stage.';
COMMENT ON SCHEMA recommendation IS 'Recommendation V ranked actions on a Hybrid CNN-BiLSTM prediction.';
COMMENT ON TABLE catalog.course IS 'A course belongs to one raw dataset (student_mat, student_por, or oulad).';
COMMENT ON TABLE catalog.enrollment IS 'Links a student to a course; predictions hang off this row.';
