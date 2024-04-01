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




    //new variables added for RL agent
    private bool canShootFire, canShootThunder, canShootEarth = true;

    private bool pressedAlready = false;

    //create an enum representing three actions: tapClick, holdClick, and noClick:

    //4 boolean variables to represent the 4 possible actions:

    private int previousAction = 5; //both should be 5 to begin.
    private int actionChoice = 0;
    private Vector2 mouseWorldPosition;

    private float mouseRotation = 0; //THIS IS JUST A TEST

    public static PlayerRL instance = null;

    [System.NonSerialized]
    public GameObject closestEnemy = null;

    private GameObject crop;

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
        // StartCoroutine(selectAction());
        // StartCoroutine(generateMousePosition());
    }

    public override void Initialize()
    {
        //do nothing
    }

    public override void OnEpisodeBegin()
    {
        water.GetComponent<ParticleSystem>().Stop();
        //do nothing
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        //Water network (2 inputs):
        GameObject cropSpawn = GameObject.Find("CropSpawn");
        int sum = 0;
        if(cropSpawn.transform.childCount > 0) {
            GameObject child = cropSpawn.transform.GetChild(0).gameObject;
            //first we think of type of crop. 
            //check if they either contain "RiceGrain", "SoybeanGrain", or "FruitGrain":
            if(child.name.Contains("RiceGrain")) {
                sensor.AddObservation(0);
                sum++;
            } else if(child.name.Contains("SoybeanGrain")) {
                sensor.AddObservation(1);
                sum++;
            } else if(child.name.Contains("FruitGrain")) {
                sensor.AddObservation(2);
                sum++;
            }
            sensor.AddObservation(child.transform.position.y);
            sum++;
            crop = child;
        } else {
            sensor.AddObservation(3); //none
            sensor.AddObservation(0);
            sum+=2;
        }

        //print out how many observations we have:
    
        //THIS IS FOR THE BASIC FIRE NETWORK.
        /*
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        //sort it by distance from player:
        System.Array.Sort(enemies, delegate(GameObject a, GameObject b) {
            return Vector2.Distance(a.transform.position, transform.position).CompareTo(Vector2.Distance(b.transform.position, transform.position));
        });
        
        int enemiesCap = 1; //JUST ONE ENEMY!
        
        if(enemies.Length > 0) {
            closestEnemy = enemies[0];
            
            
            for(int i = 0; i < enemiesCap; i++) {
                if(i < enemies.Length) {
                    //instead of x and y, we will do distance and angle:
                    // sensor.AddObservation(Vector2.Distance(enemies[i].transform.position, transform.position));

                    //we want the angle to be positive or negative: aka, directly above player would be 90, directly below would be -90.
                    // if(enemies[i].transform.position.y > transform.position.y) {
                    //     sensor.AddObservation(Vector2.Angle(enemies[i].transform.position - transform.position, Vector2.right));
                    // } else {
                    //     sensor.AddObservation(-1 * Vector2.Angle(enemies[i].transform.position - transform.position, Vector2.right));
                    // }

                    //get x distance and y distance between enemy and player, and normalize it:
                    Vector2 coordinates = enemies[i].transform.position - transform.position;
                    sensor.AddObservation(coordinates.x);
                    sensor.AddObservation(coordinates.y);
                    //then put distance from player to enemy:
                    // sensor.AddObservation(Vector2.Distance(enemies[i].transform.position, transform.position));
                } else {
                    Vector2 coordinates = enemies[enemies.Length - 1].transform.position - transform.position;  
                    // coordinates.x = (coordinates.x + 2.29f) / 11.09f;
                    // coordinates.y = (coordinates.y + 4.84f) / 9.68f;
                    sensor.AddObservation(coordinates.x);
                    sensor.AddObservation(coordinates.y);
                    // sensor.AddObservation(Vector2.Distance(enemies[enemies.Length - 1].transform.position, transform.position));
                }
            }
            //Debug.Log(coordinates);
        }
        else {
            closestEnemy = null;
            for(int i = 0; i < enemiesCap; i++) {
                //for first thing, random value between 3 and 12:
                sensor.AddObservation(0);
                //next, something random from -90 to 90:
                sensor.AddObservation(0);
            }
        }
        */
        
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        previousAction = actionChoice;
        // //this is for water network
        // actionChoice = actionBuffers.DiscreteActions[0];
        // if(actionChoice == 1) //map 1 to no op
        // {
        //     actionChoice = 4;
        // }
        mouseRotation = ScaleAction(actionBuffers.ContinuousActions[0], 90f, 270f);
        //THIS WAS FOR FIRE NETWORK:
        //mouseRotation = ScaleAction(actionBuffers.ContinuousActions[0], -180f, 180f);        

        // //water = 0, fire = 1, earth = 2, thunder = 3, nothing = 4
        // // actionChoice = actionType;
        // actionChoice = 1;
        // mouseWorldPosition = new Vector2(mouseX, mouseY);
    }

    //heuristic for testing. so discrete actions[0] will be based on w,a,s,d. continuous[0] will be the x position of the mouse, and continuous[1] will be the y position of the mouse:
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        var continuousActionsOut = actionsOut.ContinuousActions;

        //discrete actions:
        discreteActionsOut[0] = 4; //default to do nothing
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 0;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 3;
        }

        
        Vector3 vec = Camera.main.ScreenToWorldPoint(Input.mousePosition);
        continuousActionsOut[0] = vec.x;
        continuousActionsOut[1] = vec.y;
    }

    //create a public function for adding rewards to the agent:
    public void AddRewardExternal(float reward)
    {
        AddReward(reward);
    }

    // Update is called once per frame
    void Update()
    {

        //THIS IS FOR FIRE NETWORK:
        /*
        GameObject[] enemies = GameObject.FindGameObjectsWithTag("Enemy");
        System.Array.Sort(enemies, delegate(GameObject a, GameObject b) {
            return Vector2.Distance(a.transform.position, transform.position).CompareTo(Vector2.Distance(b.transform.position, transform.position));
        });
        if(enemies.Length > 0) {
            closestEnemy = enemies[0];
            Vector2 dir = closestEnemy.transform.position - transform.position;
            float ang = Vector2.Angle(dir, transform.right);
            if(ang < 5) {
                Debug.Log("FACING ENEMY");
                AddRewardExternal(1f);
            }
        }
        */


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
                    // Vector2 dir = crop.transform.position - transform.position;
                    // float ang = Vector2.Angle(dir, transform.right);
                    // if(crop.transform.position.y > transform.position.y) {
                    //     shooterAxis.rotation = Quaternion.AngleAxis(ang, Vector3.forward);
                    // } else {
                    //     shooterAxis.rotation = Quaternion.AngleAxis(-1 * ang, Vector3.forward);
                    // }
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
                Vector3 mousePos = Camera.main.ScreenToWorldPoint(mousePosition);
                mousePos.z = 0;
                Instantiate(thunderPrefab, mousePos, Quaternion.identity);
                //thunderSound.Play();
                StartCoroutine(SetCanThunder(1f));
            }
            else if (actionChoice == EARTH_INDEX && canShootEarth && rockCount < 3 && Camera.main.ScreenToWorldPoint(mousePosition).x >= -2.8f) {
                canShootEarth = false;
                Vector3 mousePos = Camera.main.ScreenToWorldPoint(mousePosition);
                mousePos.z = 0;
                Instantiate(earthPrefab, mousePos, Quaternion.identity);
                rockCount++;
                StartCoroutine(SetCanEarth(0.6f));
            }
            
            
        }

        rockCountText.text = "x " + (3 - rockCount).ToString() + "/3";
    }







    public void TakeDamage(int dmg) {
        health -= dmg;
        healthBarFill.fillAmount = (float)health / (float)maxHealth;

        // AddRewardExternal(-1f);

        
        if(health <= 0) {
            Die();
        }
    }

    public void Die() {
        GameManager.numCorrect = 0;
        GameManager.numIncorrect = 0;

        Debug.Log("Net Reward: " + GetCumulativeReward());
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
