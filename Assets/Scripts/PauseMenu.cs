using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PauseMenu : MonoBehaviour
{
    public GameObject pauseScreen;
    public bool isPaused = false;
    GameObject player;

    void Start() {
        player = GameObject.Find("Player");
    }
    

    public void loadPauseScreen() {
        if(isPaused == false) {
            PauseGame();
        } else {
            ResumeGame();
        }
    }

    public void PauseGame() {
        if(player == null) {
            player = GameObject.Find("Player");
        }
        //player component player canswitch = false
        if(player.GetComponent<Player>() != null) {
            player.GetComponent<Player>().canSwitch = false;
        } else {
            player.GetComponent<PlayerRL>().canSwitch = false;
        }
        Time.timeScale = 0;
        isPaused = true;
        pauseScreen.SetActive(true);
        //pause all audio
        AudioListener.pause = true;
    }

    public void ResumeGame() {
        if(player == null) {
            player = GameObject.Find("Player");
        }        
        Time.timeScale = 1;
        isPaused = false;
        pauseScreen.SetActive(false);
        //unpause all audio
        AudioListener.pause = false;
        if(player.GetComponent<Player>() != null) {
            player.GetComponent<Player>().canSwitch = true;
        } else {
            player.GetComponent<PlayerRL>().canSwitch = true;
        }
    }

    public void LoadMainMenu() {
        if(player.GetComponent<Player>() != null) {
            player.GetComponent<Player>().canSwitch = true;
        } else {
            player.GetComponent<PlayerRL>().canSwitch = true;
        }
        //player.GetComponent<Player>().canSwitch = true;
        
        GameManager.numCorrect = 0;
        GameManager.numIncorrect = 0;
        isPaused = false;
        Time.timeScale = 1;
        //audio fix
        AudioListener.pause = false;
        UnityEngine.SceneManagement.SceneManager.LoadScene("Title");
    }
}
