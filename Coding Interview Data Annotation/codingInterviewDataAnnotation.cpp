#include <iostream>
#include <curl/curl.h>
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

struct DataPoint {
  int x;
  int y;
  string character;
};

size_t WriteCallback(void *contents, size_t size, size_t numElements, string *data) {
  size_t totalSize = size * numElements;
  data->append((char *)contents, totalSize);
  return totalSize;
}

string extractCellText(string& html, size_t& pos) {

  size_t tdStart = html.find("<td", pos);
  if (tdStart == string::npos) {
    return "not found";
  }
  
  size_t tdTagEnd = html.find(">", tdStart);
  if (tdTagEnd == string::npos) {
    return "not found";
  }
  
  size_t tdEnd = html.find("</td>", tdTagEnd);
  if (tdEnd == string::npos) {
    return "not found";
  }
  
  string cellContent = html.substr(tdTagEnd + 1, tdEnd - tdTagEnd - 1);
  
  string text = "";
  size_t i = 0;
  while (i < cellContent.length()) {
    if (cellContent[i] == '<') {
      // Skip to end of tag
      while (i < cellContent.length() && cellContent[i] != '>') {
        i++;
      }
      i++; // Skip the >
    } else {
      text += cellContent[i];
      i++;
    }
  }
  
  // Trim whitespace
  size_t first = text.find_first_not_of(" \t\n\r");
  size_t last = text.find_last_not_of(" \t\n\r");
  if (first != string::npos && last != string::npos) {
    text = text.substr(first, last - first + 1);
  } else {
    text = "";
  }
  
  pos = tdEnd + 5; // Move past </td>
  
  return text;
}

int main()
{
    
    // This initializes the libcurl
    CURL* curl = curl_easy_init();
    if (curl) {
        string response;
        
        // Set URL
        curl_easy_setopt(curl, CURLOPT_URL,
                         "https://docs.google.com/document/d/e/"
                         "2PACX-1vRPzbNQcx5UriHSbZ-9vmsTow_R6RRe7eyAU60xIF9Dlz-"
                         "vaHiHNO2TKgDi7jy4ZpTpNqM7EvEcfr_p/pub");

        // Set callback to save the response
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback); // Function pointer for processing the response chunks
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response); // Telling libcurl where to store the parsed chunks
        
        // Perform the request
        CURLcode res = curl_easy_perform(curl);
        
        if (res == CURLE_OK) {

            // Find the table and save position
            size_t tablePos = response.find("<table");
            if (tablePos == string::npos) {
                cout << "Table not found!" << endl;
            } else {
                cout << "Found table" << endl;
                
                // Find first data row
                size_t firstRowEnd = response.find("</tr>", tablePos);
                size_t pos;
                if (firstRowEnd != string::npos) {
                    pos = firstRowEnd + 5;
                } else {
                    pos = tablePos;
                }
                
                vector<DataPoint> dataPoints;
                int maxX = 0;
                int maxY = 0;

                // Parse multiple rows
                for (int row = 0; row < 700; row++) { // Try up to 100 rows

                    size_t rowStart = response.find("<tr", pos);
                    if (rowStart == string::npos) break;
                    
                    pos = rowStart;
                    
                    string xStr = extractCellText(response, pos);
                    string charStr = extractCellText(response, pos);
                    string yStr = extractCellText(response, pos);
                    
                    if (xStr != "not found" && yStr != "not found" && !xStr.empty() && !yStr.empty()) {
                        try {
                            DataPoint point;
                            point.x = stoi(xStr);
                            if (point.x > maxX) maxX = point.x;
                            point.y = stoi(yStr);
                            if (point.y > maxY) maxY = point.y;
                            point.character = charStr;
                            dataPoints.push_back(point);
                            
                        } catch (...) {
                            // Skip if conversion fails (might be header row)
                        }
                    } else {
                        // No more valid data
                        break;
                    }
                }
                
                // Create matrix with proper dimensions (need +1 because 0-indexed)
                int rows = maxY + 1;
                int cols = maxX + 1;
                vector<vector<string>> matrix(rows, vector<string>(cols, " "));
                
                // Fill matrix (note: matrix[y][x] because y is row, x is column)
                for (size_t i = 0; i < dataPoints.size(); i++) {
                    int y = dataPoints[i].y;
                    int x = dataPoints[i].x;
                    if (!dataPoints[i].character.empty()) {
                        matrix[y][x] = dataPoints[i].character;  // Store full string, not just first char
                    }
                }
                
                for (int i = rows - 1; i >= 0; i--) {
                    for (int j = 0; j < cols; j++) {
                        cout << matrix[i][j];
                    }
                    cout << endl;
                }
            }
        } else {
            cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << endl;
        }
        
        curl_easy_cleanup(curl);
    } else {
        cerr << "Failed to initialize libcurl" << endl;
    }
    
    return 0;
}
