1.在本地创建一个仓库，写好自己的名字

2.打开你要上传的文件，点击右键，打开git_bash

3.初始化 git init,会出现.git文件，这时候所有文件就变为了红色的，表示为追踪的状态。

4.连接远程仓库，git remote add origin 你的网址（github仓库下载的地方有，在SSH里面）

5.将文件先添加至暂存区，git add 文件名（全程）或着 git add *.py（这表示将所有的.py文件提交上去）

6.将文件给提交上去，git commit -m "这次提交的备注"

7.将文件提交至github仓库，git push -u origin master 