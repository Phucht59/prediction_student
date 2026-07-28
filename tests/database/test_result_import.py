def test_three_datasets_loaded(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute("SELECT slug FROM catalog.dataset ORDER BY slug")
        assert [row[0] for row in cursor.fetchall()] == ["oulad", "student-mat", "student-por"]


def test_thirty_model_dataset_results(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT d.slug,count(*),count(*) FILTER (WHERE m.is_selected)
            FROM ml.model m JOIN catalog.dataset d USING(dataset_id)
            GROUP BY d.slug ORDER BY d.slug
            """
        )
        assert cursor.fetchall() == [
            ("oulad", 10, 1),
            ("student-mat", 10, 1),
            ("student-por", 10, 1),
        ]
