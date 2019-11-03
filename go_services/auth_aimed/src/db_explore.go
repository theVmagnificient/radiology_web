package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"github.com/lib/pq"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type RP map[string]interface{}

type ResInfo struct {
	Token 	string
	Path 	string
	Slices 	[]int64
}


func parseGetParams(r *http.Request) (string, error) {
	var tmp string
	if tmp = r.FormValue("token"); tmp == "" {
		return "", fmt.Errorf("no token found")
	} else if len(tmp) != 32 {
		return "", fmt.Errorf("bad token format")
	}

	table := strings.Split(r.URL.Path, "/")

	if table[1] != "research" {
		return "", fmt.Errorf("wrong database name")
	}

	return tmp, nil
}

func parsePutParams(r *http.Request) (*ResInfo, error) {
	params := new(ResInfo)
	decoder := json.NewDecoder(r.Body)

	table := strings.Split(r.URL.Path, "/")

	if table[1] != "research" {
		return nil, fmt.Errorf("wrong database name")
	}

	err := decoder.Decode(&params)
	if err != nil {
		return nil, err
	}

	return params, nil
}

func validatePutParams(params ResInfo) error {
	if params.Token == "" {
		return fmt.Errorf("token field cannot be empty")
	} else if len(params.Token) != 32 {
		return fmt.Errorf("bad token format")
	}

	if params.Path == "" {
		return fmt.Errorf("path field cannot be empty")
	} else if _, err := os.Stat(filepath.Join("/mnt/results", params.Path)); os.IsNotExist(err) {
		return fmt.Errorf("target folder error")
	}
	return nil
}

func putRows(db *sql.DB, params ResInfo) error {
	query := "INSERT INTO researches (token, path, pic_num) values ($1, $2, $3)"

	_, err := db.Exec(query, params.Token, params.Path, pq.Array(params.Slices))

	if err != nil {
		if err.(*pq.Error).Code == "23505" {
			return fmt.Errorf("token exists")
		} else {
			return err
		}
	}
	return nil
}

func getRows(db *sql.DB, token string) (*ResInfo, error) {

	row := db.QueryRow("SELECT token, path FROM researches WHERE token = $1", token)

	ri := new(ResInfo)

	err := row.Scan(&ri.Token, &ri.Path)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("record not found")
	} else if err != nil {
		return nil, err
	}

	// Now get pic numbers separately
	sel := "SELECT pic_num FROM researches WHERE token=$1"
	if err := db.QueryRow(sel, token).Scan(pq.Array(&ri.Slices)); err != nil {
		fmt.Println(err.Error())
		return nil, err
	}

	return ri, nil
}

func NewDbExplorer(db *sql.DB) (http.Handler, error) {

	serverMux := http.NewServeMux()

	serverMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			w.WriteHeader(http.StatusBadRequest)
			js, _ := json.Marshal(RP{"error": "database not specified"})
			w.Write(js)
			return
		}

		switch r.Method {
		case http.MethodGet:
			token, err := parseGetParams(r)
			if err != nil {
				js, _ := json.Marshal(RP{"error": err.Error()})
				w.WriteHeader(http.StatusBadRequest)
				w.Write(js)
				return
			}

			res, err := getRows(db, token)

			if err != nil {
				js, _ := json.Marshal(RP{"error": err.Error()})
				if err.Error() == "record not found" {
					w.WriteHeader(http.StatusNotFound)
				} else {
					w.WriteHeader(http.StatusInternalServerError)
				}
				w.Write(js)
				return
			}


			js, err := json.Marshal(RP{"response": RP{"record": res}})
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusOK)
			w.Write(js)
		case http.MethodPut:
			params, err := parsePutParams(r)
			if err != nil {
				js, _ := json.Marshal(RP{"error": err.Error()})
				w.WriteHeader(http.StatusBadRequest)
				w.Write(js)
				return
			}

			err = validatePutParams(*params)
			if err != nil {
				js, _ := json.Marshal(RP{"error": err.Error()})
				w.WriteHeader(http.StatusBadRequest)
				w.Write(js)
				return
			}

			err = putRows(db, *params)

			if err != nil {
				js, _ := json.Marshal(RP{"error": err.Error()})
				w.WriteHeader(http.StatusBadRequest)
				w.Write(js)
				return
			}
			js, err := json.Marshal(RP{"response": "success"})
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusOK)
			w.Write(js)
		}

	})
	return serverMux, nil
}
