# PostgreSQL — migration, recovery và kiểm tra quyền

Tài liệu này dùng khi tạo môi trường PostgreSQL mới hoặc phục hồi database. Không chạy migration destructive trên production để phục vụ test.

## 1. Vai trò kết nối

- Admin/DDL role: chỉ dùng cho backup, migration, role repair và evidence registration cần quyền cao.
- Application role: least-privileged; phải là `NOSUPERUSER`, `NOCREATEDB`, `NOCREATEROLE` và không sở hữu schema.
- Không dùng user `postgres` làm application role để làm permission test xanh giả.
- DSN/password chỉ đặt trong environment runtime; không commit `.env`, command output, dump hoặc credential.

## 2. Backup gate

Trước mọi database write có tác động:

1. Audit schema, counts, constraints, indexes, roles và dependency.
2. Tạo custom-format `pg_dump` ngoài repository.
3. Lưu hash/size/timestamp và kiểm tra dump đọc được.
4. Chạy migration dry-run trong transaction rồi `ROLLBACK`.
5. Chỉ `COMMIT` khi postcondition và dependency audit đều PASS.

## 3. Migration order

Chạy file trong `database/migrations/` theo thứ tự số. Các lớp chính:

- source/version/record và target lineage;
- experiment run, split, prediction, metric và recommendation;
- governed recommendation policies/revisions/advisor/follow-up;
- OULAD snapshot/evidence registry;
- fair ensemble evidence registry và set-based integrity triggers.

Ví dụ chạy một migration bằng admin DSN đã đặt trong environment:

```powershell
psql $env:POSTGRES_ADMIN_DSN -v ON_ERROR_STOP=1 -f database/migrations/005_oulad_lineage_and_snapshot_registry.sql
```

Không dùng `DROP DATABASE`, `DROP SCHEMA public CASCADE`, `TRUNCATE CASCADE` hoặc unbounded `DELETE` trong workflow kiểm thử.

## 4. Post-migration checks

- Không orphan source/target/split/prediction rows.
- Không duplicate prediction key hoặc evidence key.
- Completed run/evidence không update được.
- Invalid status bị constraint từ chối.
- App role đọc/ghi đúng allowlist nhưng không thể DROP/ALTER schema.
- Target tách khỏi feature payload.
- Registered artifact rows tái tạo đúng prediction, threshold và metric.

## 5. Validation commands

Validation portable, không training:

```powershell
py -3.10 scripts/validate_thesis_release.py
```

Full tests:

```powershell
py -3.10 -m pytest -q
```

Integration tests chỉ được mở khi admin/app DSN cùng trỏ tới disposable test setup và app DSN dùng least-privileged role. Nếu không có disposable DSN, test phải SKIP với waiver rõ ràng; không đổi thành PASS giả.

## 6. Recovery

Khi migration thất bại:

1. Giữ nguyên log lỗi đã redacted.
2. Rollback transaction nếu chưa commit.
3. Nếu đã commit, dùng compensating migration hoặc restore custom dump đã kiểm tra.
4. Chạy lại schema/count/orphan/permission/reproduction checks.
5. Không sửa trực tiếp immutable evidence để khớp database mới.
