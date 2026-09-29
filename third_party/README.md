# third_party

## DGPPO (đã pin)

DGPPO được nhúng làm submodule và **ghim tại commit `51b3b11`**
(repo gốc: https://github.com/MIT-REALM/dgppo). Cả nhóm phát triển trên đúng
phiên bản này — KHÔNG tự ý cập nhật submodule khi chưa thống nhất.

### Lấy code DGPPO về (bắt buộc sau khi clone repo nhóm)

Clone kèm submodule:

    git clone --recurse-submodules <url-repo-nhom>

Nếu đã lỡ clone không kèm submodule (thư mục `third_party/dgppo` rỗng):

    git submodule update --init --recursive

### Nếu cần đổi commit DGPPO (chỉ khi cả nhóm đồng ý)

    cd third_party/dgppo && git checkout <COMMIT_MOI> && cd ../..
    git add third_party/dgppo
    git commit -m "Bump DGPPO -> <COMMIT_MOI>"
