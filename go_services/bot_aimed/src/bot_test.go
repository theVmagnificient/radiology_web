package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	tb "gopkg.in/tucnak/telebot.v2"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
	"time"
)

type Case struct {
	Token  string
	Body   interface{}
	Result interface{}
}

var (
	client = &http.Client{Timeout: time.Second}
)

func TestBot(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

/*	token := os.Getenv("TELEBOT_SECRET")
	if token == "" {
		fmt.Println("ERROR: " +
			"In order to test telebot functionality, you need to set up " +
			"TELEBOT_SECRET environmental variable, which represents an API " +
			"key to a Telegram bot.\n")
		t.Fatal("Could't find TELEBOT_SECRET, aborting.")
	}
*/

	_, err := initNewBot("token.secret", "")//tb.NewBot(tb.Settings{Token: "917932392:AAErKmSFAio5G9MS77NAcT9zQuNakl0G1sU"})
	if err != nil {
		t.Fatal("couldn't create bot:", err)
	}
}

func TestRecipient(t *testing.T) {
	/*token := os.Getenv("TELEBOT_SECRET")
	if token == "" {
		fmt.Println("ERROR: " +
			"In order to test telebot functionality, you need to set up " +
			"TELEBOT_SECRET environmental variable, which represents an API " +
			"key to a Telegram bot.\n")
		t.Fatal("Could't find TELEBOT_SECRET, aborting.")
	}*/

	bot, err := initNewBot("token.secret", "")
	if err != nil {
		t.Fatal("couldn't create bot:", err)
	}

	bot.Bot.Send(&tb.User{}, "")
	bot.Bot.Send(&tb.Chat{}, "")
}

func handleApiParams(r *http.Request, w http.ResponseWriter) CR {
	token := r.FormValue("token")

	// success case
	if token == "94B8CEA57C49A3007225A0C70C475450" {
		return CR{"response":
			CR{"record":
				CR{"Token": "94B8CEA57C49A3007225A0C70C475450",
					"Path": "/res1",
					"Slices": []int{1, 2, 3, 4},
		}}}
	}
	if token == "94B8CEA57C49A3007225A0C70C475411" {
		return CR{"response":
			CR{"record":
				CR{"Token": "94B8CEA57C49A3007225A0C70C475411",
					"Path": "/res1",
					"Slices": []int{1, 2, 3, 5},
			}}}
	}
	if token == "94B8CEA57C49A3007225A0C711111111" {
		w.WriteHeader(http.StatusNotFound)
		return CR{"error": "record not found"}
	}

	if token == "1234" {
		w.WriteHeader(http.StatusBadRequest)
		return CR{"error": "bad token format"}
	}

	return CR{}
}

func TestPathGetter(t *testing.T) {
	// Start a local HTTP server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Test request parameters
		resp := handleApiParams(r, w)
		js, _ := json.Marshal(resp)
		// Send response to be tested
		w.Write(js)
	}))

	cases := []Case{
		Case{
			Token: "94B8CEA57C49A3007225A0C711111111",
			Body:	CR{"token": "94B8CEA57C49A3007225A0C711111111"},
			Result: []string{"Your record not found"},
		},
		Case{
			Token: "94B8CEA57C49A3007225A0C70C475450",
			Body:	CR{"token": "94B8CEA57C49A3007225A0C70C475450"},
			Result: []string{"success", "/mnt/results/res1/1.png", "/mnt/results/res1/3.png", "/mnt/results/res1/4.png"},
		},
		Case{
			Token: "94B8CEA57C49A3007225A0C70C475411",
			Body:	CR{"token": "94B8CEA57C49A3007225A0C70C475411"},
			Result: []string{"success", "/mnt/results/res1/1.png", "/mnt/results/res1/3.png", "/mnt/results/res1/5.png"},
		},
		Case{
			Token: "1234",
			Body:	CR{"token": "1234"},
			Result: []string{"Bad token format | Must be 32 symbol length"},
		},
	}
	b, err := initNewBot("token.secret", server.URL)

	if err != nil {
		t.Fatal("couldn't create bot:", err)
	}

	runCasesPath(t, server, b, cases)
}

func runCasesPath(t *testing.T, ts *httptest.Server, bot *botTG, cases []Case) {

	for idx, item := range cases {
		caseName := fmt.Sprintf("case %d: token = [%s]", idx, item.Token)

		result, err  := bot.getPicsPath(item.Token)

		if err != nil {
			t.Fatalf("Error: [%v]", err)
		}



		expected := item.Result

		if !reflect.DeepEqual(result, expected) {
			t.Fatalf("[%s] results not match\nGot : %#v\nWant: %#v", caseName, result, expected)
			continue
		}
	}
}

func TestExampleMsg(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Test request parameters
		resp := handleApiParams(r, w)
		js, _ := json.Marshal(resp)
		// Send response to be tested
		w.Write(js)
	}))

	cases := []Case{
		Case{
			Token: 	"1234",
			Body:	CR{"token": "1234"},
			Result: CR{"message": "Bad token format | Must be 32 symbol length"},
		},
		Case{
			Token: "94B8CEA57C49A3007225A0C70C475450",
			Body:	CR{"token": "94B8CEA57C49A3007225A0C70C475450"},
			Result: CR{"message": "photo"},
		},
		Case{
			Token: "94B8CEA57C49A3007225A0C711111111",
			Body:	CR{"token": "94B8CEA57C49A3007225A0C711111111"},
			Result: CR{"message": "Your record not found"},
		},
		Case{
			Token: 	"1234 jasdnasjkdn ajskdn a",
			Body:	CR{"token": "1234 jasdnasjkdn ajskdn a"},
			Result: CR{"message": "Bad token format | Must be 32 symbol length"},
		},
	}
	b, err := initNewBot("token.secret", server.URL)

	if err != nil {
		t.Fatal("couldn't create bot:", err)
	}
	go b.Bot.Start()
	runCasesMsg(t, server, b, cases)
}

func runCasesMsg(t *testing.T, ts *httptest.Server, bot *botTG, cases []Case) {
	flaskURL := os.Getenv("FLASK_TEST_URL")

	for idx, item := range cases {

		var (
			result   interface{}
			expected interface{}
		)

		caseName := fmt.Sprintf("case %d: token = [%s]", idx, item.Token)

		reqBody, err := json.Marshal(CR{"token": item.Token})

		if err != nil {
			t.Fatalf("Cannot marhsall json [%v]", err)
		}

		resp, err := http.Post(flaskURL + "/send", "application/json", bytes.NewReader(reqBody))

		if err != nil {
			t.Fatalf("Cannot do post [%v]", err)
		}
		defer resp.Body.Close()

		time.Sleep(time.Second * 60)

		respAns, err := http.Get(flaskURL + "/")
		if err != nil {
			t.Fatalf("Cannot do get [%v]", err)
		}
		defer respAns.Body.Close()
		body, err := ioutil.ReadAll(respAns.Body)

		data, err := json.Marshal(item.Result)
		json.Unmarshal(data, &expected)
		json.Unmarshal(body, &result)


		if !reflect.DeepEqual(result, expected) {
			t.Fatalf("[%s] results not match\nGot : %#v\nWant: %#v", caseName, result, expected)
			continue
		}
	}
}
