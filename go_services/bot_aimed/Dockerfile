# build stage
FROM golang:alpine AS build-env
RUN apk --no-cache add build-base git bzr mercurial gcc
RUN go get gopkg.in/tucnak/telebot.v2
RUN go get golang.org/x/net/proxy
COPY src /src
RUN cd /src && go build -o appbin *.go

# final stage
FROM alpine
WORKDIR /app
COPY --from=build-env /src/appbin /app/
COPY --from=build-env /src/token.secret /app/
ENTRYPOINT ./appbin
