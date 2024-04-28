using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemySpawner : MonoBehaviour
{
    public GameObject wolfPrefab;
    public GameObject oniPrefab;
    public GameObject kappaPrefab;
    public float rightX, topY, bottomY;
    public Vector2 horizontalRange, verticalRange;
    public float startSpawnDelay;
    public float endSpawnDelay;
    float spawnDelay;
    public GameObject enemyParent;
    public bool hardMode;

    public bool doLevel2Enemies = false;
    public bool doLevel3Enemies = false;

    // Start is called before the first frame update
    void Start()
    {
        spawnDelay = startSpawnDelay;
        StartCoroutine(SpawnEnemy());    
    }

    IEnumerator SpawnEnemy() {
        //over time, we will increase max num enemies allowed, and spawn delay
        while(true) {
            yield return new WaitForSeconds(spawnDelay);
            //also wait until the enemyParent has less than 20 children:
            while(enemyParent.transform.childCount >= 20) { //limit so theres not so many
                yield return new WaitForSeconds(0.1f);
            }

            //set spawnDelay between startSpawnDelay and endSpawnDelay, based on numCorrect over 30
            spawnDelay = Mathf.Lerp(startSpawnDelay, endSpawnDelay, GameManager.numCorrect - 30f);

            GameObject enemyPrefab;
            if(GameManager.numCorrect < 3 && !doLevel2Enemies && !doLevel3Enemies) {
                enemyPrefab = wolfPrefab;
            }
            else if(GameManager.numCorrect >= 3 && GameManager.numCorrect < 6 || (doLevel2Enemies && !doLevel3Enemies) ) {
                //70% chance of spawning wolf, 30% chance of spawning kappa
                if(hardMode) {
                    //chance is 50 50
                    if(Random.Range(0, 100) < 50) {
                        enemyPrefab = wolfPrefab;
                    }
                    else {
                        enemyPrefab = kappaPrefab;
                    }
                } else {
                    if(Random.Range(0, 100) < 60) {
                        enemyPrefab = wolfPrefab;
                    }
                    else {
                        enemyPrefab = kappaPrefab;
                    }
                }
                
            }
            else {
                //60% chance of spawning wolf, 20% chance of spawning kappa, 30% chance of spawning oni
                if(hardMode) {
                    int randomN = Random.Range(0, 100);
                    if(randomN < 33) {
                        enemyPrefab = wolfPrefab;
                    }
                    else if(randomN < 66) {
                        enemyPrefab = kappaPrefab;
                    }
                    else {
                        enemyPrefab = oniPrefab;
                    }
                } else {
                    int randomN = Random.Range(0, 100);
                    if(randomN < 60) {
                        enemyPrefab = wolfPrefab;
                    }
                    else if(randomN < 80) {
                        enemyPrefab = kappaPrefab;
                    }
                    else {
                        enemyPrefab = oniPrefab;
                    }
                }

            }

            //randomly generate a number between 1 to 3 inclusive
            int randomNum = Random.Range(1, 4);
            //if its 1, instantiate at the right side of the screen
            if(randomNum == 1) {
                transform.position = new Vector2(rightX, Random.Range(verticalRange.x, verticalRange.y));
            }
            //if its 2, instantiate at the top of the screen
            else if(randomNum == 2) {
                transform.position = new Vector2(Random.Range(horizontalRange.x, horizontalRange.y), topY);
            }
            //if its 3, instantiate at the bottom of the screen
            else if(randomNum == 3) {
                transform.position = new Vector2(Random.Range(horizontalRange.x, horizontalRange.y), bottomY);
            }
            //Instantiate(enemyPrefab, transform.position, Quaternion.identity);
            //instead, instantiate so parent is enemyParent
            Instantiate(enemyPrefab, transform.position, Quaternion.identity, enemyParent.transform);
        }
    }

    
}
