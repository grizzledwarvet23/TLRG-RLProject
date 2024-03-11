using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class MainMenu : MonoBehaviour
{
    public GameObject mainMenu;
    public GameObject helpMenu;

    //load game method:
    public void LoadGame()
    {
        //load the game scene:
        UnityEngine.SceneManagement.SceneManager.LoadScene("GameScene");
    }

    public void LoadHardGame() {
        UnityEngine.SceneManagement.SceneManager.LoadScene("GameSceneHard");
    }

    public void LoadTitleScene() {
        UnityEngine.SceneManagement.SceneManager.LoadScene("Title");
    }

    public void LoadHelp()
    {
        //load the help scene:
        mainMenu.SetActive(false);
        helpMenu.SetActive(true);
    }

    public void LoadMain() {
        mainMenu.SetActive(true);
        helpMenu.SetActive(false);
    }

    public void QuitGame()
    {
        //quit the game:
        Application.Quit();
    }
}
