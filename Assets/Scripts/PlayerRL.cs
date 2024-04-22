using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;    
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class PlayerRL : Agent
{
    public Transform firepoint;
    private Camera mainCamera;

    public bool canSwitch = true;

    public Transform shooterAxis;


    private int elementIndex;

    private float buttonPressStartTime;
    private float buttonPressDuration;

    private const int WATER_INDEX = 0;
    private const int FIRE_INDEX = 1;
    private const int THUNDER_INDEX = 3;
    private const int EARTH_INDEX = 2;

    public GameObject water;
    public GameObject firePrefab;
    public GameObject thunderPrefab;
    public GameObject earthPrefab;

    public GameObject rockShadow;
    public GameObject thunderShadow;


    public Sprite[] sprites;
    public SpriteRenderer hand;

    public AudioSource waterSound;

    [System.NonSerialized]
    public int rockCount;
    public TextMeshProUGUI rockCountText;

    public float lowOpacity, highOpacity;
    public Image[] elementIcons;

    public int health;
    private int maxHealth;
    public Image healthBarFill;

    private bool rewardAdded = false;




    //new variables added for RL agent
    private bool canShootFire, canShootThunder, canShootEarth = true;

    private bool pressedAlready = false;

    //create an enum representing three actions: tapClick, holdClick, and noClick:

    //4 boolean variables to represent the 4 possible actions:

    private int previousAction = 1; //both should be 5 to begin.
    private int actionChoice = 1;
    private Vector2 mouseWorldPosition;

    private float mouseRotation = 0; //THIS IS JUST A TEST

    public static PlayerRL instance = null;

    [System.NonSerialized]
    public GameObject[] closestEnemies = new GameObject[3];

    [System.NonSerialized]
    public float[] closestEnemiesX = new float[3];

    [System.NonSerialized]
    public float[] closestEnemiesY = new float[3];

    [System.NonSerialized]
    public float[] observationsArray = new float[5];

    private GameObject crop;

    private int reward = 0;

    void Awake()
    {
        if (instance == null)
        {
            instance = this;
            //do not destroy on load:\
            DontDestroyOnLoad(gameObject);
        }
        else if (instance != this)
        {
            Destroy(gameObject);
            return;
        }
    }
    // Start is called before the first frame update
    void Start()
    {
        elementIndex = 0;
        mainCamera = Camera.main;
        canShootFire = true;
        canShootThunder = true;
        canShootEarth = true;
        maxHealth = health;
        rockCountText.text = "x " + (3 - rockCount).ToString() + "/3";
        rockShadow.SetActive(false);
        thunderShadow.SetActive(false);
    }

    public int GetTopDecision()
    {
        //the hierarchically top level decision, to be used in determining. 
        //which lower policy network to activate.
        return actionChoice;
    }

    public float[] GetObservations()
    {
        return observationsArray;
        //3 enemy X's, 3 enemy Y's
        // float[] res = new float[8];
        // for(int i = 0; i < 6; i+=2)
        // {
        //     if(closestEnemies[i / 2] != null)
        //     {
        //         res[i] = closestEnemies[i / 2].transform.position.x - transform.position.x;
        //         res[i + 1] = closestEnemies[i / 2].transform.position.y - transform.position.y;
        //     } else {
        //         res[i] = 0;
        //         res[i + 1] = 0;
        //     }
        // }

        // //that was 3 closest enemies. next, we will put the crop type and crop position.
        // //type of crop (rice = 0, green = 1, orange = 2, none = 3)
        // //y position of crop (default is 0).
        // //Water network (2 inputs):

        // GameObject cropSpawn = GameObject.Find("CropSpawn");
        // if(cropSpawn.transform.childCount > 0)
        // {
        //     GameObject child = cropSpawn.transform.GetChild(0).gameObject;
        //     if(child.name.Contains("RiceGrain"))
        //     {
        //         res[6] = 0;
        //     }
        //     else if(child.name.Contains("SoybeanGrain"))
        //     {
        //         res[6] = 1;
        //     }
        //     else if(child.name.Contains("FruitGrain"))
        //     {
        //         res[6] = 2;
        //     }
        //     res[7] = child.transform.position.y;   
        // } else {
        //     res[6] = 3;
        //     res[7] = 0;
        // }
        
        // return res;
    }

    public void SetAction(int choice, int third, float rotateValue)
    {

        if(choice == 0) {
            //this is the crop sorter code.
            if(third == 0)
            {
                mouseRotation = ScaleAction(rotateValue, 90f, 150f);
            } 
            else if(third == 1)
            {
                mouseRotation = ScaleAction(rotateValue, 150f, 210f);
            }
            else 
            {
                mouseRotation = ScaleAction(rotateValue, 210f, 270f);
            }
        } 
        else if(choice == 1) //FIRE CASE!
        {
            if(third == 0)
            {
                mouseRotation = ScaleAction(rotateValue, -90f, -60f);
            } 
            else if(third == 1)
            {
                mouseRotation = ScaleAction(rotateValue, -60f, -30f);
            } 
            else if(third == 2)
            {
                mouseRotation = ScaleAction(rotateValue, -30f, 0f);
            } else if (third == 3)
            {
                mouseRotation = ScaleAction(rotateValue, 0f, 30f);
            } else if (third == 4)
            {
                mouseRotation = ScaleAction(rotateValue, 30f, 60f);
            } else if (third == 5)
            {
                mouseRotation = ScaleAction(rotateValue, 60f, 90f);
            } 
        

        }
        
        
    }

    public int GetReward() //gets the reward at this current time step.
    {
        return reward;
    }

    public void AddRewardCustom(int r)
    {
        if(!rewardAdded) 
        {
            StartCoroutine(AddRewardCoroutine(r));
        }
    }

    private IEnumerator AddRewardCoroutine(int r)
    {
        reward += r;
        rewardAdded = true;
        yield return new WaitForSeconds(0.05f); //we do this because we want this reward to have some "time" before it goes away.
        reward -= r;
        rewardAdded = false;
    }




    public override void OnEpisodeBegin()
    {
        water.GetComponent<ParticleSystem>().Stop();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //12 INPUTS RIGHT NOW!


        List<float> observationsList = new List<float>();
        // health (1 input)
        sensor.AddObservation(health);
        observationsList.Add(health);
        
        

        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        System.Array.Sort(enemies, delegate(GameObject a, GameObject b) {
            return Vector2.Distance(a.transform.position, transform.position).CompareTo(Vector2.Distance(b.transform.position, transform.position));
        });
        //set closestEnemies from that:
        for (int i = 0; i < 3; i++) {
            if(i < enemies.Length) {
                closestEnemies[i] = enemies[i];
                closestEnemiesX[i] = enemies[i].transform.position.x;
                closestEnemiesY[i] = enemies[i].transform.position.y;
            } else {
                closestEnemies[i] = null;
                closestEnemiesX[i] = 0;
                closestEnemiesY[i] = 0;
            }
        }
        
        int enemiesCap = 3; //WE SHALL TRY 3 ENEMIES!
        
        //enemies x, y, health (3 enemies rn, so 3 * 3 = 9 inputs)
        if(enemies.Length > 0) {
            for(int i = 0; i < enemiesCap; i++) {
                if(i < enemies.Length) {

                    Vector2 coordinates = enemies[i].transform.position - transform.position;
                    sensor.AddObservation(coordinates.x);
                    sensor.AddObservation(coordinates.y);
                    //add the health of the enemy.
                    sensor.AddObservation(enemies[i].GetComponent<Enemy>().health);

                    observationsList.Add(coordinates.x);
                    observationsList.Add(coordinates.y);
                    observationsList.Add(enemies[i].GetComponent<Enemy>().health);

                    
                } else {
                    Vector2 coordinates = enemies[enemies.Length - 1].transform.position - transform.position;  
                    sensor.AddObservation(coordinates.x);
                    sensor.AddObservation(coordinates.y);
                    //also add the health of the enemy.
                    sensor.AddObservation(enemies[enemies.Length - 1].GetComponent<Enemy>().health);

                    observationsList.Add(coordinates.x);
                    observationsList.Add(coordinates.y);
                    observationsList.Add(enemies[enemies.Length - 1].GetComponent<Enemy>().health);
                }
            }
        }
        else {
            
            for(int i = 0; i < enemiesCap; i++) {
                sensor.AddObservation(0);
                sensor.AddObservation(0);
                sensor.AddObservation(0);

                observationsList.Add(0);
                observationsList.Add(0);
                observationsList.Add(0);

            }
        }

        //Water network (2 inputs):
        GameObject cropSpawn = GameObject.Find("CropSpawn");
        int sum = 0;
        if(cropSpawn.transform.childCount > 0) {
            GameObject child = cropSpawn.transform.GetChild(0).gameObject;
            //first we think of type of crop. 
            //check if they either contain "RiceGrain", "SoybeanGrain", or "FruitGrain":
            if(child.name.Contains("RiceGrain")) {
                sensor.AddObservation(0);

                observationsList.Add(0);

            } else if(child.name.Contains("SoybeanGrain")) {
                sensor.AddObservation(1);

                observationsList.Add(1);
            } else if(child.name.Contains("FruitGrain")) {
                sensor.AddObservation(2);

                observationsList.Add(2);
            }
            sensor.AddObservation(child.transform.position.y);
            observationsList.Add(child.transform.position.y);
            crop = child;
        } else {
            sensor.AddObservation(3); //none
            sensor.AddObservation(0);

            observationsList.Add(3);
            observationsList.Add(0);
        }

        observationsArray = observationsList.ToArray();
                
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // previousAction = actionChoice;
        // actionChoice = actionBuffers.DiscreteActions[0];

        previousAction = actionChoice;
        actionChoice = actionBuffers.DiscreteActions[0];

        // int third = actionBuffers.DiscreteActions[0];
        // //let us just do from -90 to 90 divided into thirds:

        // float rotateValue = actionBuffers.ContinuousActions[0];
        // if(third == 0)
        // {
        //     mouseRotation = ScaleAction(rotateValue, -90f, -60f);
        // } 
        // else if(third == 1)
        // {
        //     mouseRotation = ScaleAction(rotateValue, -60f, -30f);
        // } 
        // else if(third == 2)
        // {
        //     mouseRotation = ScaleAction(rotateValue, -30f, 0f);
        // } else if (third == 3)
        // {
        //     mouseRotation = ScaleAction(rotateValue, 0f, 30f);
        // } else if (third == 4)
        // {
        //     mouseRotation = ScaleAction(rotateValue, 30f, 60f);
        // } else if (third == 5)
        // {
        //     mouseRotation = ScaleAction(rotateValue, 60f, 90f);
        // } 
        

        // previousAction = actionChoice;
        // int element = actionBuffers.DiscreteActions[0];
        // //the special check. ROCK AND FIRE CASE for SURVIVAL MODEL
        // if(element == 0) //we will say that 0 is for rock.
        // {
        //     actionChoice = 2;
        //     float mouseX = ScaleAction(actionBuffers.ContinuousActions[0], -2.76f, 8.5f);
        //     float mouseY = ScaleAction(actionBuffers.ContinuousActions[1], -4.73f, 5.03f);
        //     mouseWorldPosition = new Vector2(mouseX, mouseY);
        // } 
        // else {
        //     actionChoice = 1;
        // }
        
        // int third = actionBuffers.DiscreteActions[0];
        // float rotateValue = actionBuffers.ContinuousActions[0];
        // if(third == 0)
        // {
        //     mouseRotation = ScaleAction(rotateValue, -120f, -60f);
        // } 
        // else if(third == 1)
        // {
        //     mouseRotation = ScaleAction(rotateValue, -60f, 0f);
        // } 
        // else if(third == 2)
        // {
        //     mouseRotation = ScaleAction(rotateValue, 0f, 60f);
        // } 
        // else if (third == 3)
        // {
        //     mouseRotation = ScaleAction(rotateValue, 60f, 120f);
        // }


        // int third = actionBuffers.DiscreteActions[0];
        // float angle = actionBuffers.ContinuousActions[0];
        //we will do a thirded fire model from -120 to 120

        
        
        //a discrete action, 0 to do water option, 1 to do fire option.
        // previousAction = actionChoice;
        // actionChoice = 0;
        
        // int option = actionBuffers.DiscreteActions[0];

        // if(option == 0)
        // {
        //     actionChoice = 0;
        //     //and now, basically calling the water NN.
        // }
        // else if(option == 1)
        // {
        //     actionChoice = 1;   
        //     //and now, basically calling the fire NN.
        // }
        
        // int third = actionBuffers.DiscreteActions[0];
        // if(third == 0) //top third.
        // {
        //     mouseRotation = ScaleAction(actionBuffers.ContinuousActions[0], 90f, 150f);
        // } 
        // else if(third == 1)
        // {
        //     mouseRotation = ScaleAction(actionBuffers.ContinuousActions[0], 150f, 210f);
        // }
        // else 
        // {
        //     mouseRotation = ScaleAction(actionBuffers.ContinuousActions[0], 210f, 270f);
        // }
        

        // //this is for water network
        // actionChoice = actionBuffers.DiscreteActions[0];
        // if(actionChoice == 1) //map 1 to no op
        // {
        //     actionChoice = 4;
        // }
        // actionChoice = 0;
        // mouseRotation = ScaleAction(actionBuffers.ContinuousActions[0], 90f, 270f);
        //THIS WAS FOR FIRE NETWORK:
        // Debug.Log("action: " + actionBuffers.ContinuousActions[0]);
        // mouseRotation = ScaleAction(actionBuffers.ContinuousActions[0], -180f, 180f);        

        // //water = 0, fire = 1, earth = 2, thunder = 3, nothing = 4
        // // actionChoice = actionType;
        // actionChoice = 1;
        // mouseWorldPosition = new Vector2(mouseX, mouseY);
    }

    // //heuristic for testing. so discrete actions[0] will be based on w,a,s,d. continuous[0] will be the x position of the mouse, and continuous[1] will be the y position of the mouse:
    // public override void Heuristic(in ActionBuffers actionsOut)
    // {
    //     var discreteActionsOut = actionsOut.DiscreteActions;
    //     var continuousActionsOut = actionsOut.ContinuousActions;

    //     //discrete actions:
    //     discreteActionsOut[0] = 4; //default to do nothing
    //     if (Input.GetKey(KeyCode.W))
    //     {
    //         discreteActionsOut[0] = 0;
    //     }
    //     else if (Input.GetKey(KeyCode.A))
    //     {
    //         discreteActionsOut[0] = 1;
    //     }
    //     else if (Input.GetKey(KeyCode.S))
    //     {
    //         discreteActionsOut[0] = 2;
    //     }
    //     else if (Input.GetKey(KeyCode.D))
    //     {
    //         discreteActionsOut[0] = 3;
    //     }
    //     Vector3 vec = Camera.main.ScreenToWorldPoint(Input.mousePosition);
    //     continuousActionsOut[0] = vec.x;
    //     continuousActionsOut[1] = vec.y;
    // }

    //create a public function for adding rewards to the agent:
    public void AddRewardExternal(float reward)
    {
        AddReward(reward);
    }

    // Update is called once per frame
    void Update()
    {
        //AddRewardExternal(0.05f * (health / maxHealth));
        //we do this because we want to encourage the agent to stay alive with a higher health.
        //THIS IS FOR FIRE NETWORK:
        
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        System.Array.Sort(enemies, delegate(GameObject a, GameObject b) {
            return Vector2.Distance(a.transform.position, transform.position).CompareTo(Vector2.Distance(b.transform.position, transform.position));
        });
        if(enemies.Length > 0) {
            //based on length of enemies, set closestEnemies, closestEnemiesX, and closestEnemiesY:
            for(int i = 0; i < 3; i++) {
                if(i < enemies.Length) {
                    closestEnemies[i] = enemies[i];
                    closestEnemiesX[i] = enemies[i].transform.position.x;
                    closestEnemiesY[i] = enemies[i].transform.position.y;
                } else {
                    closestEnemies[i] = null;
                    closestEnemiesX[i] = 0;
                    closestEnemiesY[i] = 0;
                }
            }
            // closestEnemy = enemies[0];
            Vector2 dir = closestEnemies[0].transform.position - transform.position;
            float ang = Mathf.Abs(Vector2.Angle(dir, transform.right));
            if(ang < 5) {
                AddRewardCustom(1); //close aimer
            }
        }        
        
       

        // Convert the mouse position to world coordinates
        if(mainCamera == null)
        {
            mainCamera = Camera.main;
        }
        if(rockShadow == null)
        {
            rockShadow = GameObject.Find("RockShadow");
            rockShadow.SetActive(false);
        }
        if(thunderShadow == null)
        {
            thunderShadow = GameObject.Find("ThunderShadow");
            thunderShadow.SetActive(false);
        }
        if(healthBarFill == null)
        {
            healthBarFill = GameObject.Find("HealthbarFill").GetComponent<Image>();
        }

        if(rockCountText == null)
        {
            rockCountText = GameObject.Find("RockCounter").GetComponent<TextMeshProUGUI>();
        }
        Vector2 mousePosition = mainCamera.WorldToScreenPoint(mouseWorldPosition);


        // Calculate the direction from the player to the mouse cursor
        Vector2 direction = mouseWorldPosition - new Vector2(transform.position.x, transform.position.y);
        // Calculate the angle in degrees
        // float angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg;
        float angle = mouseRotation;

        // Rotate the player around the Z-axis
        shooterAxis.rotation = Quaternion.AngleAxis(angle, Vector3.forward);

        //set position of rock shadow to be at mouse position. 
        Vector3 pos = Camera.main.ScreenToWorldPoint(mousePosition);
        pos.z = 0;
        rockShadow.transform.position = pos;
        thunderShadow.transform.position = pos;

        if(Camera.main.ScreenToWorldPoint(mousePosition).x >= -2.8f && rockCount != 3) {
            rockShadow.GetComponent<SpriteRenderer>().color = new Color(1, 1, 1, 1);
        } else {
            rockShadow.GetComponent<SpriteRenderer>().color = new Color(1, 0, 0, 1);
        }


        if(actionChoice != previousAction) { //decided to switch actions.
            buttonPressStartTime = Time.time;
            pressedAlready = false;
            //first do the disabling of everything:
            water.GetComponent<ParticleSystem>().Stop();
            waterSound.Stop();
            if(actionChoice < 4) { //ie, it is not just "do nothing"
                rockShadow.SetActive(false);
                thunderShadow.SetActive(false);
                
                switch (actionChoice) {
                    //water and fire cases need not do anything special
                    case 2:
                        rockShadow.SetActive(true);
                        break;
                    case 3:
                        thunderShadow.SetActive(true);
                        break;
                    default:
                        break;
                }
                hand.sprite = sprites[actionChoice]; //sprite/visual changes.

                if(elementIcons[0] == null)
                {
                    //elementIcons[0] is game object "Water" whose parent is called "ElementIcons":
                    elementIcons[0] = GameObject.Find("WaterIcon").GetComponent<Image>();
                    elementIcons[1] = GameObject.Find("FireIcon").GetComponent<Image>();
                    elementIcons[2] = GameObject.Find("RockIcon").GetComponent<Image>();
                    elementIcons[3] = GameObject.Find("ThunderIcon").GetComponent<Image>();
                }
                for(int i = 0; i < elementIcons.Length; i++) {
                    if(i == actionChoice) {
                        elementIcons[i].color = new Color(1, 1, 1, highOpacity / 255.0f);
                    } else {
                        elementIcons[i].color = new Color(1, 1, 1, lowOpacity / 255.0f);
                    }
                }
            }

        }
        if(actionChoice < 4 && Time.time - buttonPressStartTime > 0.18f) { //ie, it is not just "do nothing." also action has been committed to long enough.
            if(actionChoice == WATER_INDEX) {
                if(crop != null) {
                    
                    shooterAxis.rotation = Quaternion.AngleAxis(angle, Vector3.forward);
                }
                // Water
                if(water.GetComponent<ParticleSystem>().isPlaying == false) {
                    water.GetComponent<ParticleSystem>().Play();
                }
                //play water sound
                if(waterSound.isPlaying == false) {
                    waterSound.Play();
                }
            } else if (actionChoice == FIRE_INDEX && canShootFire) {
                canShootFire = false;
                Instantiate(firePrefab, firepoint.position, firepoint.rotation);
                // fireSound.Play();
                StartCoroutine(SetCanFire(0.2f));
            }
            else if (actionChoice == THUNDER_INDEX && canShootThunder && !pressedAlready) {
                // Thunder
                pressedAlready = true;
                canShootThunder = false;
                //instatiate thunder at mouse position
                // mousePos.z = 0;
                Instantiate(thunderPrefab, mouseWorldPosition, Quaternion.identity);
                //thunderSound.Play();
                StartCoroutine(SetCanThunder(1f));
            }
            else if (actionChoice == EARTH_INDEX && canShootEarth && rockCount < 3 && Camera.main.ScreenToWorldPoint(mousePosition).x >= -2.8f) {
                canShootEarth = false;
                // mousePos.z = 0;
                Instantiate(earthPrefab, mouseWorldPosition, Quaternion.identity);
                rockCount++;
                StartCoroutine(SetCanEarth(0.6f));
            }
            
            
        }

        rockCountText.text = "x " + (3 - rockCount).ToString() + "/3";
    }







    public void TakeDamage(int dmg) {
        health -= dmg;
        healthBarFill.fillAmount = (float)health / (float)maxHealth;


        
        if(health <= 0) {
            AddRewardExternal(-2);
            Die();
        }
    }

    public void Die() {
        GameManager.numCorrect = 0;
        GameManager.numIncorrect = 0;

        Debug.Log("Total reward: " + GetCumulativeReward());
        EndEpisode();

        //set health back to max
        health = maxHealth;
        rockCount = 0;
        SceneManager.LoadScene("GameSceneRL");
    }

    IEnumerator ReloadScene() {
        //probably want to use this for ending episode
        yield return new WaitForSeconds(3f);
        SceneManager.LoadScene("GameSceneRL");
    }

    IEnumerator SetCanFire(float time) {
        yield return new WaitForSeconds(time);
        canShootFire = true;
    }

    IEnumerator SetCanThunder (float time) {
        yield return new WaitForSeconds(time);
        canShootThunder = true;
    }

    IEnumerator SetCanEarth (float time) {
        yield return new WaitForSeconds(time);
        canShootEarth = true;
    }
}
