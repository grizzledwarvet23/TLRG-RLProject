using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class GameManager : MonoBehaviour
{
    public static int numCorrect = 0;
    public static int numIncorrect = 0;
    public Transform spawnPoint;

    public GameObject ricePrefab;
    public GameObject beanPrefab;
    public GameObject fruitPrefab;
    
    public GameObject cropsParent;

    bool isSpawning = false;

    public static TextMeshProUGUI scoreText;

    public bool spawnCrops = true;

    //for now, lets put it at say 20 needed correct to win.
    //if  8 crops incorrect, then u lose


    //this will also take care of spawning them crops
    // Start is called before the first frame update
    void Start()
    {
        //set score text equal to UI > CropsCount
        scoreText = GameObject.Find("CropsCount").GetComponent<TextMeshProUGUI>();
        //update score text by calling UpdateScoreText()
        UpdateScoreText();
    }

    // Update is called once per frame
    void Update()
    {
        if(spawnCrops) {
            //randomly spawn one of the three at the spawn point. do not spawn another if there is already one there. instantiate with crops parent as the parent, so use crops parent to check if there is already one there
            if (cropsParent.transform.childCount == 0 && !isSpawning) {
                isSpawning = true;
                StartCoroutine(SpawnCrops());
            }
            else {
                //check if the crop in cropsParent is below the y position of -6.5. if so, destroy it and increment numIncorrect
                if (!isSpawning && cropsParent.transform.GetChild(0).position.y < -6.5f) {
                    
                    if(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name == "GameSceneRL") {
                        PlayerRL.instance.AddRewardExternal(-1f);
                    }
                    Destroy(cropsParent.transform.GetChild(0).gameObject);
                    numIncorrect++;
                    PlayWrongSound();
                }
            }
        }

        if (numCorrect >= 30) {
            //win
            numCorrect = 0;
            numIncorrect = 0;
            if(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name != "GameSceneRL") {
                UnityEngine.SceneManagement.SceneManager.LoadScene("WinScreen");
            } else {
                //PlayerRL.instance.AddRewardExternal(10f);F
                PlayerRL.instance.Die();
            }
        } else if (numIncorrect >= 5) {
            //lose
            numCorrect = 0;
            numIncorrect = 0;
            //if the current scene is called "GameSceneRL", then load "LoseScreenRL".
            //here is the code to check the current scene name:
            if(UnityEngine.SceneManagement.SceneManager.GetActiveScene().name != "GameSceneRL") {
                UnityEngine.SceneManagement.SceneManager.LoadScene("LoseScreen2");
            } else {
                //PlayerRL.instance.AddRewardExternal(-2f);
                PlayerRL.instance.Die();
            }
        }
    }

    IEnumerator SpawnCrops() {
        yield return new WaitForSeconds(1.5f);
        int rand = Random.Range(0, 3);
        GameObject spawnedObj;
            if (rand == 0) {
                spawnedObj = Instantiate(ricePrefab, spawnPoint.position, Quaternion.identity, cropsParent.transform);
            } else if (rand == 1) {
                spawnedObj = Instantiate(beanPrefab, spawnPoint.position, Quaternion.identity, cropsParent.transform);
            } else {
                spawnedObj = Instantiate(fruitPrefab, spawnPoint.position, Quaternion.identity, cropsParent.transform);
            }
            Grain grainScript = spawnedObj.GetComponent<Grain>();
            if(numCorrect >= 15) {
                grainScript.velocity = 2f * grainScript.velocity;
            } else if(numCorrect >= 6) {
                grainScript.velocity = 1.5f * grainScript.velocity;
            }


            
        isSpawning = false;
    }

    public static void UpdateScoreText() {
        scoreText.text = "crops: " + numCorrect + "/30";
    }

    public static void PlayCorrectSound() {
        //the child of this gameobject is called "CorrectSound". play its audio source
        GameObject.Find("CorrectSound").GetComponent<AudioSource>().Play();
    }
    public static void PlayWrongSound() {
        //the child of this gameobject is called "WrongSound". play its audio source
        GameObject.Find("WrongSound").GetComponent<AudioSource>().Play();
    }

}
