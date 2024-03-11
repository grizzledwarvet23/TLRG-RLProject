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

    private int previousAction = 5;
    private int actionChoice = 5;
    private Vector2 mouseWorldPosition;


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
        // StartCoroutine(selectAction());
        // StartCoroutine(generateMousePosition());
    }

    public override void Initialize()
    {
        // Debug.Log("INITIALIZE");
        //do nothing
    }

    public override void OnEpisodeBegin()
    {
        // Debug.Log("ON EPISODE BEGIN");
        //do nothing
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Debug.Log("COLLECTING OBSERVATIONS");
        sensor.AddObservation(health);
        sensor.AddObservation(rockCount);
        
        // Debug.Log(sensor);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // Debug.Log("ACTION RECEIVED");
        previousAction = actionChoice;
        int actionType = actionBuffers.DiscreteActions[0];

        //now get mouse position from continuous actions
        float mouseX = actionBuffers.ContinuousActions[0];
        float mouseY = actionBuffers.ContinuousActions[1];

        //water = 0, fire = 1, earth = 2, thunder = 3, nothing = 4
        actionChoice = actionType;
        mouseWorldPosition = new Vector2(mouseX, mouseY);
    }

    //heuristic for testing. so discrete actions[0] will be based on w,a,s,d. continuous[0] will be the x position of the mouse, and continuous[1] will be the y position of the mouse:
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Debug.Log("HEURISTIC CALLED");
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
        if(reward > 0) {
            Debug.Log("POSITIVE REWARD: " + reward);
        }
        else if(reward < 0) {
            Debug.Log("NEGATIVE REWARD: " + reward);
        }
        AddReward(reward);
    }

    // Update is called once per frame
    void Update()
    {
        RequestDecision();
        // Convert the mouse position to world coordinates

        Vector2 mousePosition = mainCamera.WorldToScreenPoint(mouseWorldPosition);


        // Calculate the direction from the player to the mouse cursor
        Vector2 direction = mouseWorldPosition - new Vector2(transform.position.x, transform.position.y);
        // Calculate the angle in degrees
        float angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg;

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
        if(health <= 0) {
            GameManager.numCorrect = 0;
            GameManager.numIncorrect = 0;
            SceneManager.LoadScene("LoseScreen");
        }
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
