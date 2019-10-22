package main

import (
	"bufio"
	"encoding/json"
	"golang.org/x/net/proxy"
	tb "gopkg.in/tucnak/telebot.v2"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"time"
)

type botTG struct {
	Bot 		*tb.Bot
	AuthURL		string
	errors 		[]error
}

type ResInfo struct {
	Path 	string
	Slices 	[]int64
	Token 	string
}

type Resp struct {

}

func initNewBot(tokenFile string, authURL string) (*botTG, error) {
	bot := new(botTG)
	var err error
	var credentials []string
	file, err := os.Open(tokenFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		credentials = append(credentials, scanner.Text())
	}

	bot.AuthURL = authURL

	dialer, err := proxy.SOCKS5("tcp", credentials[1],
		&proxy.Auth{User: credentials[2], Password: credentials[3]}, proxy.Direct)
	if err != nil {
		log.Fatal("Error creating dialer, aborting.")
	}

	httpTransport := &http.Transport{}
	httpClient := &http.Client{Transport: httpTransport}
	httpTransport.Dial = dialer.Dial

	bot.Bot, err = tb.NewBot(tb.Settings{
		Token:  credentials[0],
		Poller: &tb.LongPoller{Timeout: 10 * time.Second},
		Client: httpClient,
	})

	if err != nil {
		log.Fatal(err)
		return nil, err
	}
	bot.Bot.Handle("/start", func(m *tb.Message) {
		bot.Bot.Send(m.Sender, "Input token")
	})

	bot.Bot.Handle(tb.OnText, func(m *tb.Message) {
		str, err := bot.getPicsPath(m.Text)

		if err != nil || str == nil {
			str = make([]string, 1)
			str[0] = "Service unavailable"
		}
		if str[0] == "success" {
			//bot.Bot.Send(m.Sender, "success")
			for _, val := range str[1:] {
				photo := &tb.Photo{File: tb.FromDisk(val)}
				_, err := bot.Bot.Send(m.Sender, photo)
				if err != nil {
					bot.errors = append(bot.errors, err)
				}
			}
		} else {
			bot.Bot.Send(m.Sender, str[0])
		}

	})

	return bot, nil
}

type CR map[string]interface{}

// token from bot message here
func (t *botTG) getPicsPath(token string) ([]string, error) {

	var res ResInfo
	var tmp map[string]interface{}

	resp, err := http.Get(t.AuthURL+"?"+"token="+token)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}
		err = json.Unmarshal(body, &tmp)

		a := tmp["response"].(map[string]interface{})["record"].(map[string]interface{})

		res.Token = a["Token"].(string)
		res.Path = a["Path"].(string)
		sl := reflect.ValueOf(a["Slices"].([]interface{}))

		for i := 0; i < sl.Len(); i++ {
			res.Slices = append(res.Slices, int64(sl.Index(i).Interface().(float64)))
		}
		if err != nil {
			return nil, err
		}
		out, err := t.checkValidPaths(res)
		if err != nil {
			return nil, err
		}
		return out, nil
	} else if resp.StatusCode == http.StatusNotFound {
		return []string{"Your record not found"}, nil
	} else if resp.StatusCode == http.StatusBadRequest {
		return []string{"Bad token format | Must be 32 symbol length"}, nil
	}
	return nil, nil
}

func (t *botTG)checkValidPaths(res ResInfo) ([]string, error) {
	var out []string
	for _, val := range res.Slices {
		matches, err := filepath.Glob(filepath.Join("/mnt/results/", res.Path, strconv.Itoa(int(val)) + ".png"))
		if err != nil {
			return nil, err
		}
		if len(matches) > 0 {
			if len(out) == 0 {
				out = append(out, "success")
			}
			out = append(out, matches[0])
		}
	}
	return out, nil
}
