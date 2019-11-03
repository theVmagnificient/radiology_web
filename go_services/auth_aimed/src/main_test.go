package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
	"time"
)

var (
	client = &http.Client{Timeout: 10 * time.Second}
)

// CaseResponse
type CR map[string]interface{}

type Case struct {
	Method string // GET по-умолчанию в http.NewRequest если передали пустую строку
	Path   string
	Query  string
	Status int
	Result interface{}
	Body   interface{}
}

func PrepareTestApis(db *sql.DB) {
	qs := []string{
		`DROP TABLE IF EXISTS researches;`,

		`CREATE TABLE researches (
		token   char(32) PRIMARY KEY,
		path    varchar(255),
		pic_num integer[]
	);`,

		`INSERT INTO researches (token, path, pic_num)
		values ('d41d8cd98f00b204e9800998ecf8427e', '/test/test', '{100, 100, 33}'), 
		('a8f5f167f44f4964e6c998dee827110c', '/mda/hah', '{2, 4, 5}');`,
	}

	for _, q := range qs {
		_, err := db.Exec(q)
		if err != nil {
			panic(err)
		}
	}
}

func CleanupTestApis(db *sql.DB) {
	qs := []string{
		`DROP TABLE IF EXISTS researches;`,
	}
	for _, q := range qs {
		_, err := db.Exec(q)
		if err != nil {
			panic(err)
		}
	}
}

func TestApis(t *testing.T) {
	// Open our jsonFile
	jsonFile, err := os.Open("db.secret")
	// if we os.Open returns an error then handle it
	if err != nil {
		t.Errorf("Cannot open file with secrets: [%v]", err)
	}

	// read our opened xmlFile as a byte array.
	byteValue, _ := ioutil.ReadAll(jsonFile)

	// we initialize our Users array
	var d dbInfo

	// we unmarshal our byteArray which contains our
	// jsonFile's content into 'users' which we defined above
	json.Unmarshal(byteValue, &d)

	str := fmt.Sprintf("user=%s password=%s dbname=%s host=db_auth sslmode=disable", d.User, d.Password, d.DBname)

	db, err := sql.Open("postgres", str)

	err = db.Ping()
	if err != nil {
    fmt.Println("Cannot connect to DB")
		panic(err)
	}

	PrepareTestApis(db)

	// возможно вам будет удобно закомментировать это чтобы смотреть результат после теста
	defer CleanupTestApis(db)

	handler, err := NewDbExplorer(db)
	if err != nil {
		panic(err)
	}

	ts := httptest.NewServer(handler)

	cases := []Case{
		Case{
			Path: "/researches",
			Status: http.StatusBadRequest,
			Result: CR{
				"error": "no token found",
			},
		},
		Case{
			Path: "/researches",
			Query: "token=123123123",
			Status: http.StatusBadRequest,
			Result: CR{
				"error": "bad token format",
			},
		},
		Case{
			Path:   "/",
			Status: http.StatusBadRequest,
			Result: CR{
				"error": "database not specified",
			},
		},
		Case{
			Path:   "/research",
			Query:  "token=a8f5f112344f4964e6c998dee827110c",
			Status: http.StatusNotFound,
			Result: CR{
				"error": "record not found",
			},
		},
		Case{
			Path:   "/research",
			Query:  "token=a8f5f167f44f4964e6c998dee827110c",
			Status: http.StatusOK,
			Result: CR{
				"response": CR{
					"record": CR{
						"Token":       "a8f5f167f44f4964e6c998dee827110c",
						"Path":       "/mda/hah",
						"Slices":      []int{2, 4, 5},
					},
				},
			},
		},
		Case{
			Path:   "/research",
			Method: http.MethodPut,
			Status: http.StatusBadRequest,
			Body: CR{
				"Token":       "a8f5f167f44f4964e6c998dee827110c",
				"Path":        "/kek/lol",
				"Slices": 	   []int{1, 2, 3, 4},
			},
			Result: CR{
				"error": "target folder error",
			},
		},
		Case{
			Path:   "/research",
			Method: http.MethodPut,
			Status: http.StatusBadRequest,
			Body: CR{
				"Path":        "/kek/lol",
				"Slices": 	   []int{1, 2, 3, 4},
			},
			Result: CR{
				"error": "token field cannot be empty",
			},
		},
		Case{
			Path:   "/research",
			Method: http.MethodPut,
			Status: http.StatusBadRequest,
			Body: CR{
				"Token":       "a8f5f167f44f4964e6c998dee827110c",
				"Slices": 	   []int{1, 2, 3, 4},
			},
			Result: CR{
				"error": "path field cannot be empty",
			},
		},
		Case{
			Path:   "/qweqweqwe",
			Method: http.MethodPut,
			Status: http.StatusBadRequest,
			Body: CR{
				"Token":       "a8f5f167f44f4964e6c998dee827110c",
				"Path":        "/kek/lol",
				"Slices": 	   []int{1, 2, 3, 4},
			},
			Result: CR{
				"error": "wrong database name",
			},
		},
		Case{
			Path:   "/research",
			Method: http.MethodPut,
			Status: http.StatusBadRequest,
			Body: CR{
				"Token":       "a8f5f167f44f4964e",
				"Path":        "/kek/lol",
				"Slices": 	   []int{1, 2, 3, 4},
			},
			Result: CR{
				"error": "bad token format",
			},
		},
		Case{
			Path:   "/research",
			Method: http.MethodPut,
			Status: http.StatusBadRequest,
			Body: CR{
				"Token":       "a8f5f167f44f4964e6c998dee827110c",
				"Path":        "/res1",
				"Slices": 	   []int{1, 2, 3, 4},
			},
			Result: CR{
				"error": "token exists",
			},
		},
		Case{
			Path:   "/research",
			Method: http.MethodPut,
			Status: http.StatusOK,
			Body: CR{
				"Token":       "d8578edf8458ce06fbc5bb76a58c5ca4",
				"Path":        "/res1",
				"Slices": 	   []int{1, 2, 3, 4},
			},
			Result: CR{
				"response": "success",
			},
		},
		Case{
			Path:   "/research",
			Query:  "token=d8578edf8458ce06fbc5bb76a58c5ca4",
			Status: http.StatusOK,
			Result: CR{
				"response": CR{
					"record": CR{
						"Token":       "d8578edf8458ce06fbc5bb76a58c5ca4",
						"Path": 	   "/res1",
						"Slices":     []int{1, 2, 3, 4},
					},
				},
			},
		},
	}

	runCases(t, ts, db, cases)
}

func runCases(t *testing.T, ts *httptest.Server, db *sql.DB, cases []Case) {
	for idx, item := range cases {
		var (
			err      error
			result   interface{}
			expected interface{}
			req      *http.Request
		)

		caseName := fmt.Sprintf("case %d: [%s] %s %s", idx, item.Method, item.Path, item.Query)
		if item.Method == "" || item.Method == http.MethodGet {
			req, err = http.NewRequest(item.Method, ts.URL+item.Path+"?"+item.Query, nil)
		} else {
			data, err := json.Marshal(item.Body)
			if err != nil {
				panic(err)
			}

			reqBody := bytes.NewReader(data)
			req, err = http.NewRequest(item.Method, ts.URL+item.Path, reqBody)
			req.Header.Add("Content-Type", "application/json")
		}

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("[%s] request error: %v", caseName, err)
			continue
		}

		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)

		// fmt.Printf("[%s] body: %s\n", caseName, string(body))
		if item.Status == 0 {
			item.Status = http.StatusOK
		}

		if resp.StatusCode != item.Status {
			t.Fatalf("[%s] expected http status %v, got %v", caseName, item.Status, resp.StatusCode)
			continue
		}

		err = json.Unmarshal(body, &result)
		if err != nil {
			t.Fatalf("[%s] cant unpack json: %v", caseName, err)
			continue
		}

		// reflect.DeepEqual не работает если нам приходят разные типы
		// а там приходят разные типы (string VS interface{}) по сравнению с тем что в ожидаемом результате
		// этот маленький грязный хак конвертит данные сначала в json, а потом обратно в interface - получаем совместимые результаты
		// не используйте это в продакшен-коде - надо явно писать что ожидается интерфейс или использовать другой подход с точным форматом ответа
		data, err := json.Marshal(item.Result)
		json.Unmarshal(data, &expected)

		if !reflect.DeepEqual(result, expected) {
			t.Fatalf("[%s] results not match\nGot : %#v\nWant: %#v", caseName, result, expected)
			continue
		}
	}
}
