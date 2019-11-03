package main

import (
	"fmt"
	"log"
	"os"
)

func main() {
	b, err := initNewBot("token.secret", os.Getenv("AUTH_URL"))

	if err != nil {
		log.Fatalf("Error on init: [%v]", err)
	}

	fmt.Println("Bot started")
	b.Bot.Start()
}
