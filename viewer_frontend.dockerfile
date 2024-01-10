FROM node:18-alpine

RUN mkdir -p /workspace/chameleon/viewer/frontend/
COPY ./chameleon/viewer/frontend/ /workspace/chameleon/viewer/frontend/
WORKDIR /workspace/chameleon/viewer/frontend/
RUN npm install

CMD ["npm", "run", "dev"]
