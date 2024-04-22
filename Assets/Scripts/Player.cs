using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;    
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class Player : MonoBehaviour
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




    //fire shootrate
    private bool canShootFire, canShootThunder, canShootEarth = true;
    private bool pressedAlready = false;

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
    }

    // Update is called once per frame
    void Update()
    {
        
        // Get the mouse position in screen coordinates
        Vector3 mousePosition = Input.mousePosition;

        // Convert the mouse position to world coordinates
        Vector3 mouseWorldPosition = mainCamera.ScreenToWorldPoint(mousePosition);

        Debug.Log(mouseWorldPosition);

        // Calculate the direction from the player to the mouse cursor
        Vector3 direction = mouseWorldPosition - transform.position;

        // Calculate the angle in degrees
        float angle = Mathf.Atan2(direction.y, direction.x) * Mathf.Rad2Deg;

        // Rotate the player around the Z-axis
        shooterAxis.rotation = Quaternion.AngleAxis(angle, Vector3.forward);

        //set position of rock shadow to be at mouse position. 
        Vector3 pos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
        pos.z = 0;
        rockShadow.transform.position = pos;
        thunderShadow.transform.position = pos;

        if(Camera.main.ScreenToWorldPoint(Input.mousePosition).x >= -2.8f && rockCount != 3) {
            rockShadow.GetComponent<SpriteRenderer>().color = new Color(1, 1, 1, 1);
        } else {
            rockShadow.GetComponent<SpriteRenderer>().color = new Color(1, 0, 0, 1);
        }


        if (Input.GetKeyDown(KeyCode.Mouse0)) {
            buttonPressStartTime = Time.time;
            pressedAlready = false;
        }
        if(Input.GetKeyUp(KeyCode.Mouse0)) {
            buttonPressDuration = Time.time - buttonPressStartTime;

            if(buttonPressDuration <= 0.18f && canSwitch) { // SWITCHING MECHANIC
                if(elementIndex == WATER_INDEX) {
                    // Water
                    water.GetComponent<ParticleSystem>().Stop();
                }
                elementIndex = (elementIndex + 1) % 4;
                if(elementIndex == EARTH_INDEX) {
                    rockShadow.SetActive(true);
                } else {
                    rockShadow.SetActive(false);
                }

                //if the mouse position is less than -4.16, make the earth shadow red
                

                if(elementIndex == THUNDER_INDEX) {
                    thunderShadow.SetActive(true);
                } else {
                    thunderShadow.SetActive(false);
                }
                hand.sprite = sprites[elementIndex];
                for(int i = 0; i < elementIcons.Length; i++) {
                    if(i == elementIndex) {
                        elementIcons[i].color = new Color(1, 1, 1, highOpacity / 255.0f);
                    } else {
                        elementIcons[i].color = new Color(1, 1, 1, lowOpacity / 255.0f);
                    }
                }
            }
            
        }
        else if(Input.GetKey(KeyCode.Mouse0)) { // firing 
            if(Time.time - buttonPressStartTime > 0.18f) {
                if(elementIndex == WATER_INDEX) {
                    // Water
                    if(water.GetComponent<ParticleSystem>().isPlaying == false) {
                        water.GetComponent<ParticleSystem>().Play();
                    }
                    //play water sound
                    if(waterSound.isPlaying == false) {
                        waterSound.Play();
                    }
                } else if (elementIndex == FIRE_INDEX && canShootFire) {
                    canShootFire = false;
                    Instantiate(firePrefab, firepoint.position, firepoint.rotation);
                    // fireSound.Play();
                    StartCoroutine(SetCanFire(0.2f));
                }
                else if (elementIndex == THUNDER_INDEX && canShootThunder && !pressedAlready) {
                    // Thunder
                    pressedAlready = true;
                    canShootThunder = false;
                    //instatiate thunder at mouse position
                    Vector3 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
                    mousePos.z = 0;
                    Instantiate(thunderPrefab, mousePos, Quaternion.identity);
                    //thunderSound.Play();
                    StartCoroutine(SetCanThunder(1f));
                }
                else if (elementIndex == EARTH_INDEX && canShootEarth && rockCount < 3 && Camera.main.ScreenToWorldPoint(Input.mousePosition).x >= -2.8f) {
                    canShootEarth = false;
                    Vector3 mousePos = Camera.main.ScreenToWorldPoint(Input.mousePosition);
                    mousePos.z = 0;
                    Instantiate(earthPrefab, mousePos, Quaternion.identity);
                    rockCount++;
                    StartCoroutine(SetCanEarth(0.6f));
                }
                
                
            }
        }
        else {
            if(elementIndex == WATER_INDEX) {
                // Water
                water.GetComponent<ParticleSystem>().Stop();
                //stop water sound
                waterSound.Stop();
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
