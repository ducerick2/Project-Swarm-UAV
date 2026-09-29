# third_party

Thêm DGPPO làm submodule, pin tại 1 commit cố định (Giai đoạn 0):

    git submodule add https://github.com/MIT-REALM/dgppo third_party/dgppo
    cd third_party/dgppo && git checkout <COMMIT_HASH> && cd ../..
    git add .gitmodules third_party/dgppo && git commit -m "Pin DGPPO tại <COMMIT_HASH>"

Clone lại repo kèm submodule:  git clone --recurse-submodules <url>
Đã clone sẵn:                  git submodule update --init --recursive
