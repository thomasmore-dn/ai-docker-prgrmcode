### to get the requirements.txt file in conda env:

pip list --format=freeze > requirements.txt

---

# Docker Containers:

## Training Container:

### Build the training container

```
docker build -t transfer_train-container -f Dockerfile_train .
```

### check gpus on training container

docker run --gpus 1 -ti transfer_train-container nvidia-smi

### Run the training container

- in unix terminal:

docker run --gpus 1 -v $(pwd)/data:/home/prgrmcode/app/data -ti --name train-container transfer_train-container

- in windows command prompt:

```
docker run --gpus 1 -v "%cd%/data:/home/prgrmcode/app/data" -ti --name train-container transfer_train-container
```

-- number of gpus, -v --volume mounts first folder from local machine to the folder in docker container, -ti target image, command(python3 .py)

## Application Container:

### Build the application container

```
docker build -t transfer_app-container -f Dockerfile_app .
```

### Run the application container

```
docker run -it --gpus 1 -p 5000:5000 --name app-container transfer_app-container bash
```

---

---

# With Docker compose:

Run everything easily from docker-compose.yml file with one command

## To create and run new training and app container together:

```
docker-compose up
```

### When train and app container are up and running, you can navigate to:

- **[localhost:5000](http://localhost:5000/)**

## Create and run a new training container:

```
docker-compose up --no-deps --build transfer_train-container
```

## Create and run a new app container for hosting the application:

```
docker-compose up --build transfer_app-container
```

## To start / reuse the existing training container:

```
docker-compose start transfer_train-container
```

## To start / reuse the existing app container:

```
docker-compose start transfer_app-container
```

## If docker uses too much disk space, run:

```
docker system prune
```

---
