# docker --tag n5ng build .

FROM python:3.10.13-bookworm

WORKDIR /run

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0" ]
CMD [ "python3", "n5ng.py" ]
