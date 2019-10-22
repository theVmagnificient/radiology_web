package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	_ "github.com/lib/pq"
	"io/ioutil"
	"net/http"
	"os"
)
type dbInfo struct {
	DBname		string `json:dbname`
	User 		string `json:user`
	Password	string `json:password`
}
func main() {
	// Open our jsonFile
	jsonFile, err := os.Open("db.secret")
	// if we os.Open returns an error then handle it
	if err != nil {
		fmt.Println(err)
	}

	// read our opened xmlFile as a byte array.
	byteValue, _ := ioutil.ReadAll(jsonFile)

	// we initialize our Users array
	var db dbInfo

	// we unmarshal our byteArray `which contains our
	// jsonFile's content into 'users' which we defined above
	json.Unmarshal(byteValue, &db)

	str := fmt.Sprintf("user=%s password=%s dbname=%s host=db_auth sslmode=disable", db.User, db.Password, db.DBname)

	d, err := sql.Open("postgres", str)
	err = d.Ping()
	if err != nil {
		panic(err)
	}
	handler, err := NewDbExplorer(d)

	if err != nil {
		panic(err)
	}

	http.ListenAndServe(":8082", handler)
}
