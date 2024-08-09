docker rmi $(docker images -q) -f
docker system prune -a
