# usage: clean_node.sh {node_name}
# kill {node_name} node all

ps -ef | grep $1 | awk '{ print $2 }' | sudo xargs kill -9


ps -ef | grep user_spec | awk '{ print $2 }' | sudo xargs kill -9
